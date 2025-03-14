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

#ifndef OOO_CORE_H_
#define OOO_CORE_H_

#include <algorithm>
#include <queue>
#include <string>
#include "core.h"
#include "g_std/g_multimap.h"
#include "memory_hierarchy.h"
#include "ooo_core_recorder.h"
#include "pad.h"
#include <cassert>
#include "tage.h"
#include "legos.h"

class OOOCore : public Core {
    private:
        FilterCache* l1i;
        FilterCache* l1d;
        const uint32_t ooo_width;
        const uint32_t ooo_prf_ports;
        const uint32_t fetch_bytes_per_cycle;

        uint64_t phaseEndCycle; //next stopping point

        uint64_t curCycle; //this model is issue-centric; curCycle refers to the current issue cycle
        uint64_t regScoreboard[MAX_REGISTERS]; //contains timestamp of next issue cycles where each reg can be sourced

        BblInfo* prevBbl;

        //Record load and store addresses
        Address loadAddrs[256];
        Address storeAddrs[256];
        uint32_t loads;
        uint32_t stores;

        uint64_t lastStoreCommitCycle;
        uint64_t lastStoreAddrCommitCycle; //tracks last store addr uop, all loads queue behind it

        //LSU queues are modeled like the ROB. Surprising? Entries are grabbed in dataflow order,
        //and for ordering purposes should leave in program order. In reality they are associative
        //buffers, but we split the associative component from the limited-size modeling.
        //NOTE: We do not model the 10-entry fill buffer here; the weave model should take care
        //to not overlap more than 10 misses.
        ReorderBuffer* loadQueue;
        ReorderBuffer* storeQueue;
        // ReorderBuffer<32, 4> loadQueue;
        // ReorderBuffer<32, 4> storeQueue;

        uint32_t curCycleRFReads; //for RF read stalls
        uint32_t curCycleIssuedUops; //for uop issue limits

        //This would be something like the Atom... (but careful, the iw probably does not allow 2-wide when configured with 1 slot)
        //WindowStructure<1024, 1 /*size*/, 2 /*width*/> insWindow; //this would be something like an Atom, except all the instruction pairing business...

        //Nehalem
        WindowStructure* insWindow;
        // WindowStructure<1024, 36 /*size*/> insWindow; //NOTE: IW width is implicitly determined by the decoder, which sets the port masks according to uop type
        ReorderBuffer* rob;
        // ReorderBuffer<128, 4> rob;

        // Agner's guide says it's a 2-level pred and BHSR is 18 bits, so this is the config that makes sense;
        // in practice, this is probably closer to the Pentium M's branch predictor, (see Uzelac and Milenkovic,
        // ISPASS 2009), which get the 18 bits of history through a hybrid predictor (2-level + bimodal + loop)
        // where a few of the 2-level history bits are in the tag.
        // Since this is close enough, we'll leave it as is for now. Feel free to reverse-engineer the real thing...
        // UPDATE: Now pht index is XOR-folded BSHR. This has 6656 bytes total -- not negligible, but not ridiculous.
        // BranchPredictorPAg* branchPred;
        // BranchPredictorPAg<11, 18, 14> branchPred;

        // Tage
        BranchPredictorTage* branchPred;

        Address branchPc;  //0 if last bbl was not a conditional branch
        bool branchTaken;
        Address branchTakenNpc;
        Address branchNotTakenNpc;

        uint64_t decodeCycle;
        CycleQueue* uopQueue;  // models issue queue
        // CycleQueue<28> uopQueue;  // models issue queue

        uint64_t instrs, uops, bbls, approxInstrs, mispredBranches;

        Counter profFetchStalls, profDecodeStalls, profIssueStalls;

        // Load-store forwarding
        // Just a direct-mapped array of last store cycles to 4B-wide blocks
        // (i.e., indexed by (addr >> 2) & (FWD_ENTRIES-1))
        struct FwdEntry {
            Address addr;
            uint64_t storeCycle;
            void set(Address a, uint64_t c) {addr = a; storeCycle = c;}
        };

        #define FWD_ENTRIES 32  // 2 lines, 16 4B entries/line
        FwdEntry fwdArray[FWD_ENTRIES];

        OOOCoreRecorder cRec;

    public:
        OOOCore(FilterCache* _l1i, FilterCache* _l1d, const OOOParams& oo_params, g_string& _name);

        void initStats(AggregateStat* parentStat);

        uint64_t getInstrs() const;
        uint64_t getPhaseCycles() const;
        uint64_t getCycles() const {return cRec.getUnhaltedCycles(curCycle);}

        void contextSwitch(int32_t gid);

        virtual void join();
        virtual void leave();

        InstrFuncPtrs GetFuncPtrs();

        // Contention simulation interface
        inline EventRecorder* getEventRecorder() {return cRec.getEventRecorder();}
        void cSimStart();
        void cSimEnd();

    private:
        inline void load(Address addr);
        inline void store(Address addr);

        /* NOTE: Analysis routines cannot touch curCycle directly, must use
         * advance() for long jumps or insWindow.advancePos() for 1-cycle
         * jumps.
         *
         * UPDATE: With decodeCycle, this difference is more serious. ONLY
         * cSimStart and cSimEnd should call advance(). advance() is now meant
         * to advance the cycle counters in the whole core in lockstep.
         */
        inline void advance(uint64_t targetCycle);

        // Predicated loads and stores call this function, gets recorded as a 0-cycle op.
        // Predication is rare enough that we don't need to model it perfectly to be accurate (i.e. the uops still execute, retire, etc), but this is needed for correctness.
        inline void predFalseLoad();
        inline void predFalseStore();

        inline void branch(Address pc, bool taken, Address takenNpc, Address notTakenNpc);

        inline void bbl(Address bblAddr, BblInfo* bblInfo, THREADID tid);

        static void LoadFunc(THREADID tid, ADDRINT addr);
        static void StoreFunc(THREADID tid, ADDRINT addr);
        static void PredLoadFunc(THREADID tid, ADDRINT addr, BOOL pred);
        static void PredStoreFunc(THREADID tid, ADDRINT addr, BOOL pred);
        static void BblFunc(THREADID tid, ADDRINT bblAddr, BblInfo* bblInfo);
        static void BranchFunc(THREADID tid, ADDRINT pc, BOOL taken, ADDRINT takenNpc, ADDRINT notTakenNpc);
} ATTR_LINE_ALIGNED;  // Take up an int number of cache lines

#endif  // OOO_CORE_H_
