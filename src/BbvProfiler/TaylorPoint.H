#ifndef TaylorPoint_INST_H
#define TaylorPoint_INST_H

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "pin.H"
#include "pad.h"

#include "Profile.H"
#include "BbTrace.H"

struct BblInfo;

struct BBIdICount {
    uint32_t id;
    uint64_t icount;
    friend std::ostream& operator<<(std::ostream& os, const BBIdICount& rhs) {
        os << rhs.id;
        return os;
    }
    bool operator==(const BBIdICount &rhs) const {
        return id == rhs.id;
    }
    bool operator<(const BBIdICount &rhs) const {
        return id < rhs.id;
    }
};

class BASICBLOCK_BUFFER
{
public:
    BASICBLOCK_BUFFER(int size, int order): _maxsize(size), _size(0), _order(order), _once_full(false)
    {
        ASSERT(_maxsize > _order, "BASICBLOCK_BUFFER size must be larger than order!");
        _buffer = new BblInfo*[_maxsize];
    }
    ~BASICBLOCK_BUFFER()
    {
        delete[] _buffer;
    }
    bool push(BblInfo* item)
    {
        _buffer[_size] = item;
        _size++;
        if(_buffer[0] == NULL){
            fprintf(stderr,"_size:%d\n", _size);
        }
        ASSERT(item!=NULL, "item is NULL");
        ASSERT(_buffer[0] != NULL,"ggg");
        return _size == _maxsize;
    }
    int size() const
    {
        return _size;
    }
    bool full() const
    {
        return _size == _maxsize;
    }
    BblInfo* at(int index) const
    {
        return _buffer[index];
    }
    int maxsize() const
    {
        return _maxsize;
    }
    void Reset()
    {
        _once_full = true;
        _size = 0;
    }
    bool once_full() const{
        return _once_full;
    }

private:
    BblInfo** _buffer;
    int _maxsize;
    int _size;
    int _order;
    bool _once_full;

};

class TaylorPoint
{
public:
    TaylorPoint(int order, bool emit_first, bool emit_last, int64_t sz, std::string outputdir);

    ~TaylorPoint();

    void PushBbl(BblInfo* bbl, THREADID tid);
    
    void DumpTbl(THREADID tid);
    
    void Finish(THREADID tid);
    
    void EmitProgramEndPerOrder(int order, THREADID tid);
    
    void FlushBasicBlockBuffer(THREADID tid);
    
    void FillTaylorBlockMap(THREADID tid);
    
    void EmitSliceEnd(THREADID tid);
    
    void EmitSliceEndPerOrder(int order, THREADID tid);
    
    void EmitProgramEnd(THREADID tid);

public:
    BbTrace<BBIdICount> bbTrace;
    BASICBLOCK_BUFFER** _basicblock_buffer;

    int _order;
    uint32_t _nthreads;
    uint64_t slice_size;
    uint32_t Pid;
    PROFILE** profiles;
    /*knobs*/
    bool EmitFirstSlice;
    bool EmitLastSlice;
    std::string OutputFile;
} ATTR_LINE_ALIGNED;  // Take up an int number of cache lines


#endif
