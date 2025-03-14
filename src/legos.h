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

#ifndef LEGOS_H_
#define LEGOS_H_

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


class FilterCache;

class WindowStructure {
    private:
        // NOTE: Nehalem has POPCNT, but we want this to run reasonably fast on Core2's, so let's keep track of both count and mask.
        struct WinCycle {
            uint8_t occUnits;
            uint8_t count;
            inline void set(uint8_t o, uint8_t c) {occUnits = o; count = c;}
        };

        WinCycle* curWin;
        WinCycle* nextWin;
        typedef g_map<uint64_t, WinCycle> UBWin;
        typedef typename UBWin::iterator UBWinIterator;
        UBWin ubWin;
        uint32_t occupancy;  // elements scheduled in the future

        uint32_t curPos;

        uint8_t lastPort;
        const uint32_t H;
        const uint32_t WSZ;

    public:
        WindowStructure(uint32_t _H, uint32_t _WSZ) : H(_H), WSZ(_WSZ) {
            curWin = gm_calloc<WinCycle>(H);
            nextWin = gm_calloc<WinCycle>(H);
            curPos = 0;
            occupancy = 0;
        }


        void schedule(uint64_t& curCycle, uint64_t& schedCycle, uint8_t portMask, uint32_t extraSlots = 0) {
            if (!extraSlots) {
                scheduleInternal<true, false>(curCycle, schedCycle, portMask);
            } else {
                scheduleInternal<true, true>(curCycle, schedCycle, portMask);
                uint64_t extraSlotCycle = schedCycle+1;
                uint8_t extraSlotPortMask = 1 << lastPort;
                // This is not entirely accurate, as an instruction may have been scheduled already
                // on this port and we'll have a non-contiguous allocation. In practice, this is rare.
                for (uint32_t i = 0; i < extraSlots; i++) {
                    scheduleInternal<false, false>(curCycle, extraSlotCycle, extraSlotPortMask);
                    // info("extra slot %d allocated on cycle %ld", i, extraSlotCycle);
                    extraSlotCycle++;
                }
            }
            assert(occupancy <= WSZ);
        }

        inline void advancePos(uint64_t& curCycle) {
            occupancy -= curWin[curPos].count;
            curWin[curPos].set(0, 0);
            curPos++;
            curCycle++;

            if (curPos == H) {  // rebase
                // info("[%ld] Rebasing, curCycle=%ld", curCycle/H, curCycle);
                std::swap(curWin, nextWin);
                curPos = 0;
                uint64_t nextWinHorizon = curCycle + 2*H;  // first cycle out of range

                if (!ubWin.empty()) {
                    UBWinIterator it = ubWin.begin();
                    while (it != ubWin.end() && it->first < nextWinHorizon) {
                        uint32_t nextWinPos = it->first - H - curCycle;
                        assert_msg(nextWinPos < H, "WindowStructure: ubWin elem exceeds limit cycle=%ld curCycle=%ld nextWinPos=%d", it->first, curCycle, nextWinPos);
                        nextWin[nextWinPos] = it->second;
                        // info("Moved %d events from unbounded window, cycle %ld (%d cycles away)", it->second, it->first, it->first - curCycle);
                        it++;
                    }
                    ubWin.erase(ubWin.begin(), it);
                }
            }
        }

        void longAdvance(uint64_t& curCycle, uint64_t targetCycle) {
            assert(curCycle <= targetCycle);

            // Drain IW
            while (occupancy && curCycle < targetCycle) {
                advancePos(curCycle);
            }

            if (occupancy) {
                // info("advance: window not drained at %ld, %d uops left", curCycle, occupancy);
                assert(curCycle == targetCycle);
            } else {
                // info("advance: window drained at %ld, jumping to %ld", curCycle, targetCycle);
                assert(curCycle <= targetCycle);
                curCycle = targetCycle;  // with zero occupancy, we can just jump to it
            }
        }

        // Poisons a range of cycles; used by the LSU to apply backpressure to the IW
        void poisonRange(uint64_t curCycle, uint64_t targetCycle, uint8_t portMask) {
            uint64_t startCycle = curCycle;  // curCycle should not be modified...
            uint64_t poisonCycle = curCycle;
            while (poisonCycle < targetCycle) {
                scheduleInternal<false, false>(curCycle, poisonCycle, portMask);
            }
            // info("Poisoned port mask %x from %ld to %ld (tgt %ld)", portMask, curCycle, poisonCycle, targetCycle);
            assert(startCycle == curCycle);
        }

    private:
        template <bool touchOccupancy, bool recordPort>
        void scheduleInternal(uint64_t& curCycle, uint64_t& schedCycle, uint8_t portMask) {
            // If the window is full, advance curPos until it's not
            while (touchOccupancy && occupancy == WSZ) {
                advancePos(curCycle);
            }

            uint32_t delay = (schedCycle > curCycle)? (schedCycle - curCycle) : 0;

            // Schedule, progressively increasing delay if we cannot find a slot
            uint32_t curWinPos = curPos + delay;
            while (curWinPos < H) {
                if (trySchedule<touchOccupancy, recordPort>(curWin[curWinPos], portMask)) {
                    schedCycle = curCycle + (curWinPos - curPos);
                    break;
                } else {
                    curWinPos++;
                }
            }
            if (curWinPos >= H) {
                uint32_t nextWinPos = curWinPos - H;
                while (nextWinPos < H) {
                    if (trySchedule<touchOccupancy, recordPort>(nextWin[nextWinPos], portMask)) {
                        schedCycle = curCycle + (nextWinPos + H - curPos);
                        break;
                    } else {
                        nextWinPos++;
                    }
                }
                if (nextWinPos >= H) {
                    schedCycle = curCycle + (nextWinPos + H - curPos);
                    UBWinIterator it = ubWin.lower_bound(schedCycle);
                    while (true) {
                        if (it == ubWin.end()) {
                            WinCycle wc = {0, 0};
                            bool success = trySchedule<touchOccupancy, recordPort>(wc, portMask);
                            assert(success);
                            ubWin.insert(std::pair<uint64_t, WinCycle>(schedCycle, wc));
                        } else if (it->first != schedCycle) {
                            WinCycle wc = {0, 0};
                            bool success = trySchedule<touchOccupancy, recordPort>(wc, portMask);
                            assert(success);
                            ubWin.insert(it /*hint, makes insert faster*/, std::pair<uint64_t, WinCycle>(schedCycle, wc));
                        } else {
                            if (!trySchedule<touchOccupancy, recordPort>(it->second, portMask)) {
                                // Try next cycle
                                it++;
                                schedCycle++;
                                continue;
                            }  // else scheduled correctly
                        }
                        break;
                    }
                    // info("Scheduled event in unbounded window, cycle %ld", schedCycle);
                }
            }
            if (touchOccupancy) occupancy++;
        }

        template <bool touchOccupancy, bool recordPort>
        inline uint8_t trySchedule(WinCycle& wc, uint8_t portMask) {
            static_assert(!(recordPort && !touchOccupancy), "Can't have recordPort and !touchOccupancy");
            if (touchOccupancy) {
                uint8_t availMask = (~wc.occUnits) & portMask;
                if (availMask) {
                    // info("PRE: occUnits=%x portMask=%x availMask=%x", wc.occUnits, portMask, availMask);
                    uint8_t firstAvail = __builtin_ffs(availMask) - 1;
                    // NOTE: This is not fair across ports. I tried round-robin scheduling, and there is no measurable difference
                    // (in our case, fairness comes from following program order)
                    if (recordPort) lastPort = firstAvail;
                    wc.occUnits |= 1 << firstAvail;
                    wc.count++;
                    // info("POST: occUnits=%x count=%x firstAvail=%d", wc.occUnits, wc.count, firstAvail);
                }
                return availMask;
            } else {
                // This is a shadow req, port has only 1 bit set
                uint8_t availMask = (~wc.occUnits) & portMask;
                wc.occUnits |= portMask;  // or anyway, no conditionals
                return availMask;
            }
        }
};

class ReorderBuffer {
    private:
        uint64_t* buf;
        uint64_t curRetireCycle;
        uint32_t curCycleRetires;
        uint32_t idx;
        const uint32_t SZ;
        const uint32_t W;

    public:
        ReorderBuffer(uint32_t _SZ, uint32_t _W) : SZ(_SZ), W(_W) {
            buf = gm_calloc<uint64_t>(SZ);
            for (uint32_t i = 0; i < SZ; i++) buf[i] = 0;
            idx = 0;
            curRetireCycle = 0;
            curCycleRetires = 1;
        }

        inline uint64_t minAllocCycle() {
            return buf[idx];
        }

        inline void markRetire(uint64_t minRetireCycle) {
            if (minRetireCycle <= curRetireCycle) {  // retire with bundle
                if (curCycleRetires == W) {
                    curRetireCycle++;
                    curCycleRetires = 0;
                } else {
                    curCycleRetires++;
                }

                /* No branches version (careful, width should be power of 2...)
                 * curRetireCycle += curCycleRetires/W;
                 * curCycleRetires = (curCycleRetires + 1) % W;
                 *  NOTE: After profiling, version with branch seems faster
                 */
            } else {  // advance
                curRetireCycle = minRetireCycle;
                curCycleRetires = 1;
            }

            buf[idx++] = curRetireCycle;
            if (idx == SZ) idx = 0;
        }
};

// Similar to ReorderBuffer, but must have in-order allocations and retires (--> faster)
class CycleQueue {
    private:
        uint64_t* buf;
        uint32_t idx;
        const uint32_t SZ;

    public:
        CycleQueue(uint32_t _SZ) : SZ(_SZ) {
            buf = gm_calloc<uint64_t>(SZ);
            for (uint32_t i = 0; i < SZ; i++) buf[i] = 0;
            idx = 0;
        }

        inline uint64_t minAllocCycle() {
            return buf[idx];
        }

        inline void markLeave(uint64_t leaveCycle) {
            //assert(buf[idx] <= leaveCycle);
            buf[idx++] = leaveCycle;
            if (idx == SZ) idx = 0;
        }
};

struct BblInfo;

struct OOOParams {
    uint32_t width;
    uint32_t prf_ports;
    uint32_t load_queue_cap;
    uint32_t store_queue_cap;
    uint32_t rob_cap;
    uint32_t ins_win_cap;
    uint32_t issue_queue_cap;
    uint32_t PAg_bhr_log_cap;
    uint32_t PAg_bhr_bits;
    uint32_t PAg_pht_log_cap;
    uint32_t tage_num_tables;
    uint32_t tage_index_size;
    uint32_t fetch_bytes_per_cycle;
};

#endif  // LEGOS_H_
