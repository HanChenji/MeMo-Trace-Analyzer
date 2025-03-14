#include <stdint.h>
#include <unordered_map>
#include "core.h"

class BasicBlockKey
{
    friend bool operator<(const BasicBlockKey& p1, const BasicBlockKey& p2)
    {
        return p1.addr < p2.addr;
    }
    friend bool operator==(const BasicBlockKey& p1, const BasicBlockKey& p2)
    {
        return p1.addr == p2.addr && p1.instrs == p2.instrs;
    }

  public:
    BasicBlockKey(uint64_t _addr, uint64_t _instrs) : addr(_addr), instrs(_instrs) {};

    uint64_t hash() const 
    {
        return addr ^ (instrs << 32); 
    }

    const uint64_t addr;
    const uint64_t instrs;
};

struct BasicBlockHash
{
    std::size_t operator()(const BasicBlockKey& k) const
    {
        return k.hash();
    }
};

typedef std::unordered_map<BasicBlockKey, BblInfo*, BasicBlockHash> BasicBlockMap;
