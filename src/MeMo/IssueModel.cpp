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

#include "IssueModel.h"
#include <algorithm>
#include <queue>
#include <string>
#include "bithacks.h"
#include "decoder.h"
#include "filter_cache.h"
#include "zsim.h"

#define DEBUG_MSG(args...)

// Stages --- more or less matched to Westmere, but have not seen detailed pipe diagrams anywhare
#define FETCH_STAGE 1
#define DECODE_STAGE 4  // NOTE: Decoder adds predecode delays to decode
#define ISSUE_STAGE 7
#define DISPATCH_STAGE 13  // RAT + ROB + RS, each is easily 2 cycles

#define L1D_LAT 4  // fixed, and FilterCache does not include L1 delay

/* global */
extern int64_t  interval_size;
extern uint64_t interval_pcount;
extern uint64_t interval_icount;
extern uint64_t total_pcount;
extern uint64_t total_icount;

IssueModel::IssueModel(const OOOParams& ooo_params, g_string& _name) : Core(_name), ooo_width(ooo_params.width), ooo_prf_ports(ooo_params.prf_ports), cRec(0, _name) {
    decodeCycle = DECODE_STAGE;  // allow subtracting from it
    curCycle = 0;
    phaseEndCycle = zinfo->phaseLength;

    for (uint32_t i = 0; i < MAX_REGISTERS; i++) {
        regScoreboard[i] = 0;
    }
    prevBbl = nullptr;
    curCycleRFReads = 0;
    curCycleIssuedUops = 0;

    instrs = 0;

    // initilize instruction window
    insWindow = gm_memalign<WindowStructure>(CACHE_LINE_BYTES);
    insWindow = new (insWindow) WindowStructure(8192, ooo_params.ins_win_cap);
    // initialize load queue
    loadQueue  = gm_memalign<ReorderBuffer>(CACHE_LINE_BYTES);
    loadQueue  = new (loadQueue) ReorderBuffer(ooo_params.load_queue_cap , ooo_params.width);
    storeQueue = gm_memalign<ReorderBuffer>(CACHE_LINE_BYTES);
    storeQueue = new (storeQueue) ReorderBuffer(ooo_params.store_queue_cap, ooo_params.width);
    // initialize reorder buffer
    rob = gm_memalign<ReorderBuffer>(CACHE_LINE_BYTES);
    rob = new (rob) ReorderBuffer(ooo_params.rob_cap, ooo_params.width);
    // initialize uop queue
    uopQueue = gm_memalign<CycleQueue>(CACHE_LINE_BYTES);
    uopQueue = new (uopQueue) CycleQueue(ooo_params.issue_queue_cap);
}

void IssueModel::initStats(AggregateStat* parentStat) {
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
    profIssueStalls = new Counter();
    profIssueStalls->init("issueStalls",  "Issue stalls");  

    coreStat->append(cyclesStat);
    coreStat->append(cCyclesStat);
    coreStat->append(icountStat);
    coreStat->append(pcountStat);
    coreStat->append(profIssueStalls);

    parentStat->append(coreStat);
}

uint64_t IssueModel::getInstrs() const {return instrs;}
uint64_t IssueModel::getPhaseCycles() const {return curCycle % zinfo->phaseLength;}

void IssueModel::contextSwitch(int32_t gid) {
    if (gid == -1) {
        // Do not execute previous BBL, as we were context-switched
        prevBbl = nullptr;
    }
}


InstrFuncPtrs IssueModel::GetFuncPtrs() {return {LoadFunc, StoreFunc, BblFunc, BranchFunc, PredLoadFunc, PredStoreFunc, FPTR_ANALYSIS, {0}};}

inline void IssueModel::bbl(Address bblAddr, BblInfo* bblInfo, THREADID tid) {
    if (!prevBbl) {
        // This is the 1st BBL since scheduled, nothing to simulate
        prevBbl = bblInfo;
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
        decodeCycle = MAX(decodeCycle + decDiff, uopQueue->minAllocCycle());
        if (decodeCycle > curCycle) {
            uint32_t cdDiff = decodeCycle - curCycle;
            curCycleIssuedUops = 0;
            curCycleRFReads = 0;
            for (uint32_t i = 0; i < cdDiff; i++) insWindow->advancePos(curCycle);
        }
        prevDecCycle = uop->decCycle;
        uopQueue->markLeave(curCycle);

        // Implement issue width limit --- we can only issue 4 uops/cycle
        if (curCycleIssuedUops >= ooo_width) {
            profIssueStalls->inc();
            curCycleIssuedUops = 0;
            curCycleRFReads = 0;
            insWindow->advancePos(curCycle);
        }
        curCycleIssuedUops++;

        // Kill dependences on invalid register
        // Using curCycle saves us two unpredictable branches in the RF read stalls code
        regScoreboard[0] = curCycle;

        uint64_t c0 = regScoreboard[uop->rs[0]];
        uint64_t c1 = regScoreboard[uop->rs[1]];

        // RF read stalls
        // if srcs are not available at issue time, we have to go thru the RF
        curCycleRFReads += ((c0 < curCycle)? 1 : 0) + ((c1 < curCycle)? 1 : 0);
        if (curCycleRFReads > ooo_prf_ports) {
            curCycleRFReads -= ooo_prf_ports;
            curCycleIssuedUops = 0;  // or 1? that's probably a 2nd-order detail
            insWindow->advancePos(curCycle);
        }

        uint64_t c2 = rob->minAllocCycle();
        uint64_t c3 = curCycle;

        uint64_t cOps = MAX(c0, c1);

        // Model RAT + ROB + RS delay between issue and dispatch
        uint64_t dispatchCycle = MAX(cOps, MAX(c2, c3) + (DISPATCH_STAGE - ISSUE_STAGE));

        // NOTE: Schedule can adjust both cur and dispatch cycles
        insWindow->schedule(curCycle, dispatchCycle, uop->portMask, uop->extraSlots);

        // If we have advanced, we need to reset the curCycle counters
        if (curCycle > c3) {
            curCycleIssuedUops = 0;
            curCycleRFReads = 0;
        }

        uint64_t commitCycle;

        // LSU simulation
        // NOTE: Ever-so-slightly faster than if-else if-else if-else
        switch (uop->type) {
            case UOP_GENERAL:
                commitCycle = dispatchCycle + uop->lat;
                break;

            case UOP_LOAD:
                {
                    // dispatchCycle = MAX(loadQueue.minAllocCycle(), dispatchCycle);
                    uint64_t lqCycle = loadQueue->minAllocCycle();
                    if (lqCycle > dispatchCycle) {
                        dispatchCycle = lqCycle;
                    }

                    // Wait for all previous store addresses to be resolved
                    dispatchCycle = MAX(lastStoreAddrCommitCycle+1, dispatchCycle);

                    Address addr = loadAddrs[loadIdx++];
                    uint64_t reqSatisfiedCycle = dispatchCycle + L1D_LAT;

                    // Enforce st-ld forwarding
                    uint32_t fwdIdx = (addr>>2) & (FWD_ENTRIES-1);
                    if (fwdArray[fwdIdx].addr == addr) {
                        /* Take the MAX (see FilterCache's code) Our fwdArray
                         * imposes more stringent timing constraints than the
                         * l1d, b/c FilterCache does not change the line's
                         * availCycle on a store. This allows FilterCache to
                         * track per-line, not per-word availCycles.
                         */
                        reqSatisfiedCycle = MAX(reqSatisfiedCycle, fwdArray[fwdIdx].storeCycle);
                    }

                    commitCycle = reqSatisfiedCycle;
                    loadQueue->markRetire(commitCycle);
                }
                break;

            case UOP_STORE:
                {
                    // dispatchCycle = MAX(storeQueue.minAllocCycle(), dispatchCycle);
                    uint64_t sqCycle = storeQueue->minAllocCycle();
                    if (sqCycle > dispatchCycle) {
                        dispatchCycle = sqCycle;
                    }

                    // Wait for all previous store addresses to be resolved (not just ours :))
                    dispatchCycle = MAX(lastStoreAddrCommitCycle+1, dispatchCycle);

                    Address addr = storeAddrs[storeIdx++];
                    uint64_t reqSatisfiedCycle = dispatchCycle + L1D_LAT;

                    // Fill the forwarding table
                    fwdArray[(addr>>2) & (FWD_ENTRIES-1)].set(addr, reqSatisfiedCycle);

                    commitCycle = reqSatisfiedCycle;
                    lastStoreCommitCycle = MAX(lastStoreCommitCycle, reqSatisfiedCycle);
                    storeQueue->markRetire(commitCycle);
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
                // force future load serialization
                lastStoreAddrCommitCycle = MAX(commitCycle, MAX(lastStoreAddrCommitCycle, lastStoreCommitCycle + uop->lat));
        }

        // Mark retire at ROB
        rob->markRetire(commitCycle);

        // Record dependences
        regScoreboard[uop->rd[0]] = commitCycle;
        regScoreboard[uop->rd[1]] = commitCycle;
    }

    instrs += bblInstrs;
    assert(instrs == total_pcount);

    if(interval_icount >= (uint64_t)interval_size) {
        cerr << "interval_icount: " << interval_icount << " total_icount: " << total_icount <<endl;
        zinfo -> periodicStatsBackend -> dump(false);// flushes trace writer
        interval_icount = 0;
        interval_pcount = 0;
    }
}

// Timing simulation code
void IssueModel::join() {
    DEBUG_MSG("[%s] Joining, curCycle %ld phaseEnd %ld", name.c_str(), curCycle, phaseEndCycle);
    uint64_t targetCycle = cRec.notifyJoin(curCycle);
    if (targetCycle > curCycle) advance(targetCycle);
    phaseEndCycle = zinfo->globPhaseCycles + zinfo->phaseLength;
    DEBUG_MSG("[%s] Joined, curCycle %ld phaseEnd %ld", name.c_str(), curCycle, phaseEndCycle);
}

void IssueModel::leave() {
    DEBUG_MSG("[%s] Leaving, curCycle %ld phaseEnd %ld", name.c_str(), curCycle, phaseEndCycle);
    cRec.notifyLeave(curCycle);
}

void IssueModel::cSimStart() {
    uint64_t targetCycle = cRec.cSimStart(curCycle);
    assert(targetCycle >= curCycle);
    if (targetCycle > curCycle) advance(targetCycle);
}

void IssueModel::cSimEnd() {
    uint64_t targetCycle = cRec.cSimEnd(curCycle);
    assert(targetCycle >= curCycle);
    if (targetCycle > curCycle) advance(targetCycle);
}

void IssueModel::advance(uint64_t targetCycle) {
    assert(targetCycle > curCycle);
    decodeCycle += targetCycle - curCycle;
    insWindow->longAdvance(curCycle, targetCycle);
    curCycleRFReads = 0;
    curCycleIssuedUops = 0;
    assert(targetCycle == curCycle);
    /* NOTE: Validation with weave mems shows that not advancing internal cycle
     * counters in e.g., the ROB does not change much; consider full-blown
     * rebases though if weave models fail to validate for some app.
     */
}

// Pin interface code

void IssueModel::LoadFunc(THREADID tid, ADDRINT addr) {}
void IssueModel::StoreFunc(THREADID tid, ADDRINT addr) {}
void IssueModel::PredLoadFunc(THREADID tid, ADDRINT addr, BOOL pred) {}
void IssueModel::PredStoreFunc(THREADID tid, ADDRINT addr, BOOL pred) {}

void IssueModel::BblFunc(THREADID tid, ADDRINT bblAddr, BblInfo* bblInfo) {
    IssueModel* core = static_cast<IssueModel*>(cores[tid]);
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

void IssueModel::BranchFunc(THREADID tid, ADDRINT pc, BOOL taken, ADDRINT takenNpc, ADDRINT notTakenNpc) {}