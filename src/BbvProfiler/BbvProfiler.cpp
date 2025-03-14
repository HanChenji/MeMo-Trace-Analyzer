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

#include "BbvProfiler.h"
#include "TaylorPoint.H"
#include "zsim.h"

extern TaylorPoint *taylorpoint;

/* global */
extern int64_t  interval_size;
extern uint64_t interval_pcount;
extern uint64_t interval_icount;
extern uint64_t total_pcount;
extern uint64_t total_icount;

BbvProfiler::BbvProfiler(g_string& _name) : Core(_name), instrs(0), bbls(0), curCycle(0), haltedCycles(0), prevBbl(nullptr) {
}

void BbvProfiler::initStats(AggregateStat* parentStat) {
    AggregateStat* coreStat = new AggregateStat();

    coreStat->init(name.c_str(), "Core stats");

    ProxyStat* icountStat = new ProxyStat();
    icountStat->init("icount", "Simulated instructions", &total_icount);
    ProxyStat* pcountStat = new ProxyStat();
    pcountStat->init("pcount", "Simulated instructions", &total_pcount);
    ProxyStat* bblsStat = new ProxyStat();
    bblsStat->init("bbls", "Basic blocks", &bbls);

    coreStat->append(icountStat);
    coreStat->append(pcountStat);
    coreStat->append(bblsStat);

    parentStat->append(coreStat);
}

uint64_t BbvProfiler::getPhaseCycles() const {
    return curCycle % zinfo->phaseLength;
}

void BbvProfiler::bbl(BblInfo* bblInfo, THREADID tid) {
    //info("BBL %s %p", name.c_str(), bblInfo);
    //info("%d %d", bblInfo->instrs, bblInfo->bytes);
    if (!prevBbl) {
        // This is the 1st BBL since scheduled, nothing to simulate
        prevBbl = bblInfo;
        return;
    }

    assert(taylorpoint != nullptr);

    /* Simulate execution of previous BBL */
    taylorpoint -> PushBbl(prevBbl, tid);

    instrs += prevBbl->instrs;
    curCycle += prevBbl->instrs;
    bbls++;
    assert(instrs == total_pcount);

    prevBbl = bblInfo;

    if(interval_icount >= (uint64_t)interval_size) {
        cerr << "interval_icount: " << interval_icount << " total_icount: " << total_icount <<endl;
        taylorpoint -> DumpTbl(tid);
        zinfo -> periodicStatsBackend -> dump(false);// flushes trace writer
        interval_icount = 0;
        interval_pcount = 0; 
    }
}

void BbvProfiler::contextSwitch(int32_t gid) {
    if (gid == -1) {
        // Do not execute previous BBL, as we were context-switched
        prevBbl = nullptr;
    }
}

void BbvProfiler::join() {
    //info("[%s] Joining, curCycle %ld phaseEnd %ld haltedCycles %ld", name.c_str(), curCycle, phaseEndCycle, haltedCycles);
    if (curCycle < zinfo->globPhaseCycles) { //carry up to the beginning of the phase
        haltedCycles += (zinfo->globPhaseCycles - curCycle);
        curCycle = zinfo->globPhaseCycles;
    }
    phaseEndCycle = zinfo->globPhaseCycles + zinfo->phaseLength;
    //note that with long events, curCycle can be arbitrarily larger than phaseEndCycle; however, it must be aligned in current phase
    //info("[%s] Joined, curCycle %ld phaseEnd %ld haltedCycles %ld", name.c_str(), curCycle, phaseEndCycle, haltedCycles);
}


//Static class functions: Function pointers and trampolines

InstrFuncPtrs BbvProfiler::GetFuncPtrs() {
    return {LoadFunc, StoreFunc, BblFunc, BranchFunc, PredLoadFunc, PredStoreFunc, FPTR_ANALYSIS, {0}};
}

void BbvProfiler::BblFunc(THREADID tid, ADDRINT bblAddr, BblInfo* bblInfo) {
    BbvProfiler* core = static_cast<BbvProfiler*>(cores[tid]);
    core->bbl(bblInfo, tid);

    while (core->curCycle > core->phaseEndCycle) {
        assert(core->phaseEndCycle == zinfo->globPhaseCycles + zinfo->phaseLength);
        core->phaseEndCycle += zinfo->phaseLength;

        uint32_t cid = getCid(tid);
        //NOTE: TakeBarrier may take ownership of the core, and so it will be used by some other thread. If TakeBarrier context-switches us,
        //the *only* safe option is to return inmmediately after we detect this, or we can race and corrupt core state. If newCid == cid,
        //we're not at risk of racing, even if we were switched out and then switched in.
        uint32_t newCid = TakeBarrier(tid, cid);
        if (newCid != cid) break; /*context-switch*/
    }
}

