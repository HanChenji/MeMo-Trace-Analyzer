#include "TaylorPoint.H"
#include "BbTrace.H"
#include "core.h"
#include "log.h"
#include "BasicBlockMap.h"

KNOB<bool> KnobTaylorPid(KNOB_MODE_WRITEONCE, "pintool",
    "taylorpoint:pid", "0", "record pid");

KNOB<std::string> KnobTaylorLengthFile(KNOB_MODE_APPEND, "pintool",
    "taylorpoint:length_file", "", "length file name");

KNOB<uint32_t> KnobTaylorNumThreads(KNOB_MODE_WRITEONCE, "pintool",
    "taylorpoint:num_threads", "1", "number of threads");

extern uint64_t interval_icount;
extern uint64_t total_icount;

extern BasicBlockMap bbl_map;

TaylorPoint::TaylorPoint(int order, bool emit_first, bool emit_last, int64_t sz, std::string outputfile)
    : bbTrace(order + 1), _order(order),
      _nthreads(KnobTaylorNumThreads.Value()),
      slice_size(sz),
      EmitFirstSlice(emit_first),
      EmitLastSlice(emit_last),
      OutputFile(outputfile)
{
    /* dump the paramters*/
    info("[TaylorPoint]: parameters");
    info("order: %d", _order);
    info("nthreads: %d", _nthreads);
    info("slice_size: %ld", slice_size);
    info("EmitFirstSlice: %d", EmitFirstSlice);
    info("EmitLastSlice: %d", EmitLastSlice);
    info("OutputFile: %s", OutputFile.c_str());

    _basicblock_buffer = new BASICBLOCK_BUFFER*[_nthreads];
    for(uint32_t i = 0; i < _nthreads; i++)
    {
        _basicblock_buffer[i] = new BASICBLOCK_BUFFER(slice_size>>3, _order);
    }

    ASSERTX(_nthreads < PIN_MAX_THREADS);
    if (KnobTaylorPid.Value())
    {
        Pid = getpid();
    }

    profiles = new PROFILE*[_nthreads];
    for(THREADID tid = 0; tid < _nthreads; tid++)
    {
        profiles[tid] = new PROFILE(_order);
    }

}

TaylorPoint::~TaylorPoint()
{
    for(uint32_t i = 0; i < _nthreads; i++)
    {
        delete _basicblock_buffer[i];
        delete profiles[i];
    }
    delete[] _basicblock_buffer;
    delete[] profiles;
}

void TaylorPoint::PushBbl(BblInfo* bbl, THREADID tid)
{
    ASSERT(bbl != NULL,"basicblock is NULL");
    if(_basicblock_buffer[tid]->push(bbl))
    {
        FlushBasicBlockBuffer(tid);
    }
}

void TaylorPoint::DumpTbl(THREADID tid)
{
    FlushBasicBlockBuffer(tid);
    EmitSliceEnd(tid);
}

#include "zsim.h"

void TaylorPoint::Finish(THREADID tid)
{
    if(EmitLastSlice && interval_icount != slice_size)
    {
        cerr << "Emitting last slice" << endl;
        DumpTbl(tid);
    }
    EmitProgramEnd(tid);
}

void TaylorPoint::EmitProgramEnd(THREADID tid)
{
    /*Emit info for the taylor blocks until _order*/
    for(int order=0; order<=_order; order++){
        EmitProgramEndPerOrder(order, tid);
    }

    /*Emit info for all the basic blocks*/
    gzofstream& bb_dump_file = profiles[tid]->BbFile[0];
    bb_dump_file << "Start Dumping BasicBlock!" << std::endl;
    bb_dump_file << "BasicBlock id " <<  "StartAddr " << "static instructions " << std::endl;
    for (auto bi: bbl_map) {
        bb_dump_file << bi.second->bblIdx;
        bb_dump_file << " " << std::hex << bi.first.addr;
        bb_dump_file << " " << std::dec << bi.second->instrs;
        bb_dump_file << std::endl;
    }
    bb_dump_file << "Stop Dumping BasicBlock!" << std::endl;

    profiles[tid] -> CloseAll();
}

void TaylorPoint::EmitProgramEndPerOrder(int order, THREADID tid)
{
    gzofstream& BbFile = profiles[tid]->BbFile[order];
    BbFile << "Cumulative icount " << std::dec
           << total_icount
           << std::endl;
    BbFile << "SliceSize: " << std::dec << slice_size << std::endl;
    BbFile << "Start Dumping TaylorBlock!" << std::endl;
    BbFile << "TaylorBlockId " << "BblIDs " << "static instructions " << "CumuBlockCount: " << std::endl;

    int idx = 1;
    auto flags = BbFile.flags();
    bbTrace.Foreach(order, [&BbFile, &idx, &order, this](Node<BBIdICount> &node) {
        BbFile << idx << " ";
        bbTrace.DumpNodeTrace(order, node, BbFile);
        BbFile << " " << node.key.icount << " " << node.count.past << std::endl;
        idx++;
    });
    BbFile.flags(flags);
    BbFile << "End Dumping TaylorBlock!" << std::endl;
    BbFile << "TaylorBlockSize Order" << std::dec << order << ": " << bbTrace.Size(order) << std::endl;
    info("TaylorBlockSize Order%d: %ld", order, bbTrace.Size(order));
}


void TaylorPoint::FlushBasicBlockBuffer(THREADID tid)
{
    FillTaylorBlockMap(tid);
    _basicblock_buffer[tid]->Reset();
}

void TaylorPoint::FillTaylorBlockMap(THREADID tid)
{
    /* scan the _basicblock_buffer and fill into the TaylorBlockMap */
    BASICBLOCK_BUFFER *cur_buffer = _basicblock_buffer[tid];
    if (cur_buffer->size() < _order + 1) {
        return;
    }

    int ptr = 0;
    if (!cur_buffer->once_full()) {
        for (; ptr <= _order; ptr++) {
            BblInfo *bbl    = cur_buffer->at(ptr);
            uint32_t id     = bbl->bblIdx;
            uint64_t icount = bbl->instrs;
            bbTrace.Init(ptr, {id, icount});
        }
    }
    while (ptr < cur_buffer->size()) {
        BblInfo *bbl    = cur_buffer->at(ptr);
        uint32_t id     = bbl->bblIdx;
        uint64_t icount = bbl->instrs;
        bbTrace.Walk({id, icount});
        ptr++;
    }
}

void TaylorPoint::EmitSliceEnd(THREADID tid)
{
    static uint64_t counter = 0;
    if (counter++ % 100 == 0) info("Executing Slice: %lu", counter);

    bbTrace.Propagate();

    for(int order=0; order<=_order; order++)
    {
        EmitSliceEndPerOrder(order, tid);
    }
    profiles[tid]->first = false;
}

void TaylorPoint::EmitSliceEndPerOrder(int order, THREADID tid)
{
    if(profiles[tid]->first && !EmitFirstSlice)
    {
        cerr << "Skipping first slice" << endl;
        return;
    }

    gzofstream &bbFile = profiles[tid]->BbFile[order];
    bbFile << "# Slice ending at " << total_icount << std::endl;
    bbFile << "T";
    int idx = 0;
    bbTrace.Foreach(order, [order, &bbFile, &idx, this](Node<BBIdICount> &node) {
        idx++;
        if (node.count.cur > 0) {
            bbFile << ":" << std::dec << idx << ":" << std::dec << node.key.icount * node.count.cur << " ";
            node.count.Rotate();
        }
    });
    bbFile << std::endl;

    bbFile.flush();
}
