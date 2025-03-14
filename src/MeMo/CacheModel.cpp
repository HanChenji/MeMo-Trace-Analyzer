/** $lic$
 * Copyright (C) 2012-2015 by Massachusetts Institute of Technology
 * Copyright (C) 2010-2013 by The Board of Trustees of Stanford University
 *
 * This file is part of zsim.
 *
 * zsim is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, version 2.
 *
 * If you use this software in your research, we request that you reference
 * the zsim paper ("ZSim: Fast and Accurate Microarchitectural Simulation of
 * Thousand-Core Systems", Sanchez and Kozyrakis, ISCA-40, June 2013) as the
 * source of the simulator in any publications that use this software, and that
 * you send us a citation of your work.
 *
 * zsim is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "CacheModel.h"
#include <algorithm>
#include <queue>
#include <string>
#include "bithacks.h"
#include "decoder.h"
#include "filter_cache.h"
#include "zsim.h"

/* Uncomment to induce backpressure to the IW when the load/store buffers fill up. In theory, more detailed,
 * but sometimes much slower (as it relies on range poisoning in the IW, potentially O(n^2)), and in practice
 * makes a negligible difference (ROB backpressures).
 */
#define LSU_IW_BACKPRESSURE

#define DEBUG_MSG(args...)
//#define DEBUG_MSG(args...) info(args)

// Core parameters
// TODO(dsm): Make OOOCore templated, subsuming these

// Stages --- more or less matched to Westmere, but have not seen detailed pipe diagrams anywhare
#define FETCH_STAGE 1
#define DECODE_STAGE 4  // NOTE: Decoder adds predecode delays to decode
#define ISSUE_STAGE 7
#define DISPATCH_STAGE 13  // RAT + ROB + RS, each is easily 2 cycles

#define L1I_LAT 3  
#define L1D_LAT 4  

/* global */
extern int64_t  interval_size;
extern uint64_t interval_pcount;
extern uint64_t interval_icount;
extern uint64_t total_pcount;
extern uint64_t total_icount;

CacheModel::CacheModel(FilterCache* _l1d, const OOOParams& ooo_params, g_string& _name) : Core(_name), l1d(_l1d), ooo_width(ooo_params.width), ooo_prf_ports(ooo_params.prf_ports), cRec(0, _name) {
    decodeCycle = DECODE_STAGE;  // allow subtracting from it
    curCycle = 0;
    phaseEndCycle = zinfo->phaseLength;

    for (uint32_t i = 0; i < MAX_REGISTERS; i++) {
        regScoreboard[i] = 0;
    }
    prevBbl = nullptr;

    lastStoreCommitCycle = 0;
    lastStoreAddrCommitCycle = 0;

    instrs = 0;

    // assert(ooo_params.rob_cap     >= ooo_params.ins_win_cap);
    // assert(ooo_params.ins_win_cap >= ooo_params.issue_queue_cap);
    // assert(ooo_params.ins_win_cap >= ooo_params.load_queue_cap);
    // assert(ooo_params.ins_win_cap >= ooo_params.store_queue_cap);
    // assert(ooo_params.issue_queue_cap >= ooo_params.width);
}

void CacheModel::initStats(AggregateStat* parentStat) {
    AggregateStat* coreStat = new AggregateStat();
    coreStat->init(name.c_str(), "Core stats");

    auto x = [this]() { return cRec.getUnhaltedCycles(curCycle); };
    LambdaStat<decltype(x)>* cyclesStat = new LambdaStat<decltype(x)>(x);
    cyclesStat->init("cycles", "Simulated unhalted cycles");

    auto y = [this]() { return cRec.getContentionCycles(); };
    LambdaStat<decltype(y)>* cCyclesStat = new LambdaStat<decltype(y)>(y);
    cCyclesStat->init("cCycles", "Cycles due to contention stalls");

    ProxyStat* pcountStat = new ProxyStat();
    pcountStat->init("pcount", "Simulated instructions", &total_pcount);
    ProxyStat* icountStat = new ProxyStat();
    icountStat->init("icount", "Simulated instructions", &total_icount);

    coreStat->append(cyclesStat);
    coreStat->append(cCyclesStat);
    coreStat->append(icountStat);
    coreStat->append(pcountStat);

    parentStat->append(coreStat);
}

uint64_t CacheModel::getInstrs() const {return instrs;}
uint64_t CacheModel::getPhaseCycles() const {return curCycle % zinfo->phaseLength;}

void CacheModel::contextSwitch(int32_t gid) {
    if (gid == -1) {
        // Do not execute previous BBL, as we were context-switched
        prevBbl = nullptr;

        // Invalidate virtually-addressed filter caches
        l1d->contextSwitch();
    }
}


InstrFuncPtrs CacheModel::GetFuncPtrs() {return {LoadFunc, StoreFunc, BblFunc, BranchFunc, PredLoadFunc, PredStoreFunc, FPTR_ANALYSIS, {0}};}

inline void CacheModel::load(Address addr) {
    loadAddrs[loads++] = addr;
}

void CacheModel::store(Address addr) {
    storeAddrs[stores++] = addr;
}

// Predicated loads and stores call this function, gets recorded as a 0-cycle op.
// Predication is rare enough that we don't need to model it perfectly to be accurate (i.e. the uops still execute, retire, etc), but this is needed for correctness.
void CacheModel::predFalseLoad() {
    loadAddrs[loads++] = -1L;
}

void CacheModel::predFalseStore() {
    storeAddrs[stores++] = -1L;
}

inline void CacheModel::bbl(Address bblAddr, BblInfo* bblInfo, THREADID tid) {
    if (!prevBbl) {
        // This is the 1st BBL since scheduled, nothing to simulate
        prevBbl = bblInfo;
        // Kill lingering ops from previous BBL
        loads = stores = 0;
        return;
    }

    /* Simulate execution of previous BBL */
    uint32_t bblInstrs = prevBbl->instrs;
    DynBbl* bbl = &(prevBbl->oooBbl[0]);
    prevBbl = bblInfo;

    uint32_t loadIdx = 0;
    uint32_t storeIdx = 0;

    uint32_t prevDecCycle = 0;

    // Run dispatch/IW
    for (uint32_t i = 0; i < bbl->uops; i++) {
        DynUop* uop = &(bbl->uop[i]);

        // Decode stalls
        uint32_t decDiff = uop->decCycle - prevDecCycle;
        decodeCycle = decodeCycle + decDiff;
        curCycle = MAX(curCycle, decodeCycle);
        prevDecCycle = uop->decCycle;

        // Kill dependences on invalid register
        // Using curCycle saves us two unpredictable branches in the RF read stalls code
        regScoreboard[0] = curCycle;

        uint64_t c0 = regScoreboard[uop->rs[0]];
        uint64_t c1 = regScoreboard[uop->rs[1]];

        uint64_t c2 = curCycle;
        uint64_t c3 = curCycle;

        uint64_t cOps = MAX(c0, c1);

        // Model RAT + ROB + RS delay between issue and dispatch
        uint64_t dispatchCycle = MAX(cOps, MAX(c2, c3) + (DISPATCH_STAGE - ISSUE_STAGE));

        curCycle = MAX(curCycle, dispatchCycle);

        uint64_t commitCycle;

        // LSU simulation
        // NOTE: Ever-so-slightly faster than if-else if-else if-else
        switch (uop->type) {
            case UOP_GENERAL:
                commitCycle = dispatchCycle + uop->lat;
                break;

            case UOP_LOAD:
                {
                    // Wait for all previous store addresses to be resolved
                    dispatchCycle = MAX(lastStoreAddrCommitCycle+1, dispatchCycle);

                    Address addr = loadAddrs[loadIdx++];
                    uint64_t reqSatisfiedCycle = dispatchCycle;
                    if (addr != ((Address)-1L)) {
                        reqSatisfiedCycle = l1d->load(addr, dispatchCycle) + L1D_LAT;
                        cRec.record(curCycle, dispatchCycle, reqSatisfiedCycle);
                    }

                    commitCycle = reqSatisfiedCycle;
                }
                break;

            case UOP_STORE:
                {
                    // Wait for all previous store addresses to be resolved (not just ours :))
                    dispatchCycle = MAX(lastStoreAddrCommitCycle+1, dispatchCycle);

                    Address addr = storeAddrs[storeIdx++];
                    uint64_t reqSatisfiedCycle = l1d->store(addr, dispatchCycle) + L1D_LAT;
                    cRec.record(curCycle, dispatchCycle, reqSatisfiedCycle);

                    commitCycle = reqSatisfiedCycle;
                    lastStoreCommitCycle = MAX(lastStoreCommitCycle, reqSatisfiedCycle);
                }
                break;

            case UOP_STORE_ADDR:
                commitCycle = dispatchCycle + uop->lat;
                lastStoreAddrCommitCycle = MAX(lastStoreAddrCommitCycle, commitCycle);
                break;

            //case UOP_FENCE:  //make gcc happy
            default:
                assert((UopType) uop->type == UOP_FENCE);
                commitCycle = dispatchCycle + uop->lat;
                // info("%d %ld %ld", uop->lat, lastStoreAddrCommitCycle, lastStoreCommitCycle);
                // force future load serialization
                lastStoreAddrCommitCycle = MAX(commitCycle, MAX(lastStoreAddrCommitCycle, lastStoreCommitCycle));
                // info("%d %ld %ld X", uop->lat, lastStoreAddrCommitCycle, lastStoreCommitCycle);
        }

        // Record dependences
        regScoreboard[uop->rd[0]] = commitCycle;
        regScoreboard[uop->rd[1]] = commitCycle;

        //info("0x%lx %3d [%3d %3d] -> [%3d %3d]  %8ld %8ld %8ld %8ld", bbl->addr, i, uop->rs[0], uop->rs[1], uop->rd[0], uop->rd[1], decCycle, c3, dispatchCycle, commitCycle);
    }

    instrs += bblInstrs;
    assert(instrs == total_pcount);

    // Check full match between expected and actual mem ops
    // If these assertions fail, most likely, something's off in the decoder
    assert_msg(loadIdx == loads, "%s: loadIdx(%d) != loads (%d)", name.c_str(), loadIdx, loads);
    assert_msg(storeIdx == stores, "%s: storeIdx(%d) != stores (%d)", name.c_str(), storeIdx, stores);
    loads = stores = 0;


    /* Simulate frontend for branch pred + fetch of this BBL
     *
     * NOTE: We assume that the instruction length predecoder and the IQ are
     * weak enough that they can't hide any ifetch or bpred stalls. In fact,
     * predecoder stalls are incorporated in the decode stall component (see
     * decoder.cpp). So here, we compute fetchCycle, then use it to adjust
     * decodeCycle.
     */

    // Model fetch-decode delay (fixed, weak predec/IQ assumption)
    uint64_t fetchCycle = decodeCycle - (DECODE_STAGE - FETCH_STAGE);
    uint32_t lineSize = 1 << lineBits;

    // Simulate current bbl ifetch
    Address endAddr = bblAddr + bblInfo->bytes;
    for (Address fetchAddr = bblAddr; fetchAddr < endAddr; fetchAddr += lineSize) {
        // The Nehalem frontend fetches instructions in 16-byte-wide accesses.
        // Do not model fetch throughput limit here, decoder-generated stalls already include it
        // We always call fetches with curCycle to avoid upsetting the weave
        // models (but we could move to a fetch-centric recorder to avoid this)
        uint64_t fetchLat = L1I_LAT;
        cRec.record(curCycle, curCycle, curCycle + fetchLat);
        fetchCycle += fetchLat;
    }

    // If fetch rules, take into account delay between fetch and decode;
    // If decode rules, different BBLs make the decoders skip a cycle
    decodeCycle++;
    uint64_t minFetchDecCycle = fetchCycle + (DECODE_STAGE - FETCH_STAGE);
    if (minFetchDecCycle > decodeCycle) {
        decodeCycle = minFetchDecCycle;
    }

    if(interval_icount >= (uint64_t)interval_size) {
        cerr << "interval_icount: " << interval_icount << " total_icount: " << total_icount <<endl;
        zinfo -> periodicStatsBackend -> dump(false);// flushes trace writer
        interval_icount = 0;
        interval_pcount = 0;
    }
}

// Timing simulation code
void CacheModel::join() {
    DEBUG_MSG("[%s] Joining, curCycle %ld phaseEnd %ld", name.c_str(), curCycle, phaseEndCycle);
    uint64_t targetCycle = cRec.notifyJoin(curCycle);
    if (targetCycle > curCycle) advance(targetCycle);
    phaseEndCycle = zinfo->globPhaseCycles + zinfo->phaseLength;
    // assert(targetCycle <= phaseEndCycle);
    DEBUG_MSG("[%s] Joined, curCycle %ld phaseEnd %ld", name.c_str(), curCycle, phaseEndCycle);
}

void CacheModel::leave() {
    DEBUG_MSG("[%s] Leaving, curCycle %ld phaseEnd %ld", name.c_str(), curCycle, phaseEndCycle);
    cRec.notifyLeave(curCycle);
}

void CacheModel::cSimStart() {
    uint64_t targetCycle = cRec.cSimStart(curCycle);
    assert(targetCycle >= curCycle);
    if (targetCycle > curCycle) advance(targetCycle);
}

void CacheModel::cSimEnd() {
    uint64_t targetCycle = cRec.cSimEnd(curCycle);
    assert(targetCycle >= curCycle);
    if (targetCycle > curCycle) advance(targetCycle);
}

void CacheModel::advance(uint64_t targetCycle) {
    assert(targetCycle > curCycle);
    decodeCycle += targetCycle - curCycle;
    curCycle = targetCycle;
    /* NOTE: Validation with weave mems shows that not advancing internal cycle
     * counters in e.g., the ROB does not change much; consider full-blown
     * rebases though if weave models fail to validate for some app.
     */
}

// Pin interface code

void CacheModel::LoadFunc(THREADID tid, ADDRINT addr) {static_cast<CacheModel*>(cores[tid])->load(addr);}
void CacheModel::StoreFunc(THREADID tid, ADDRINT addr) {static_cast<CacheModel*>(cores[tid])->store(addr);}

void CacheModel::PredLoadFunc(THREADID tid, ADDRINT addr, BOOL pred) {
    CacheModel* core = static_cast<CacheModel*>(cores[tid]);
    if (pred) core->load(addr);
    else core->predFalseLoad();
}

void CacheModel::PredStoreFunc(THREADID tid, ADDRINT addr, BOOL pred) {
    CacheModel* core = static_cast<CacheModel*>(cores[tid]);
    if (pred) core->store(addr);
    else core->predFalseStore();
}

void CacheModel::BblFunc(THREADID tid, ADDRINT bblAddr, BblInfo* bblInfo) {
    CacheModel* core = static_cast<CacheModel*>(cores[tid]);
    core->bbl(bblAddr, bblInfo, tid);

    while (core->curCycle > core->phaseEndCycle) {
        core->phaseEndCycle += zinfo->phaseLength;

        uint32_t cid = getCid(tid);
        // NOTE: TakeBarrier may take ownership of the core, and so it will be used by some other thread. If TakeBarrier context-switches us,
        // the *only* safe option is to return inmmediately after we detect this, or we can race and corrupt core state. However, the information
        // here is insufficient to do that, so we could wind up double-counting phases.
        uint32_t newCid = TakeBarrier(tid, cid);
        // NOTE: Upon further observation, we cannot race if newCid == cid, so this code should be enough.
        // It may happen that we had an intervening context-switch and we are now back to the same core.
        // This is fine, since the loop looks at core values directly and there are no locals involved,
        // so we should just advance as needed and move on.
        if (newCid != cid) break;  /*context-switch, we do not own this context anymore*/
    }
}

void CacheModel::BranchFunc(THREADID tid, ADDRINT pc, BOOL taken, ADDRINT takenNpc, ADDRINT notTakenNpc) {}