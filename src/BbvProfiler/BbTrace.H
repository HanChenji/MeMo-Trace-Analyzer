#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>
#include <map>
#include <cassert>
#include <chrono>

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_constructive_interference_size;
using std::hardware_destructive_interference_size;
#else
// 64 bytes on x86-64 │ L1_CACHE_BYTES │ L1_CACHE_SHIFT │ __cacheline_aligned │ ...
constexpr std::size_t hardware_constructive_interference_size = 64;
constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

struct 
alignas(hardware_constructive_interference_size)
BbCounter
{
    uint64_t cur{};
    uint64_t past{};

    uint64_t Incement() {
        return ++cur;
    }

    uint64_t Add(uint64_t addend) {
        return cur += addend;
    }

    uint64_t Rotate() {
        past += cur;
        uint64_t tmp = cur;
        cur = 0;
        return tmp;
    }

    uint64_t Total() {
        return cur + past;
    }
};

#define FAST_MAXN 4UL
template<class T>
struct Node {
    T key;
    BbCounter count;

    size_t parent;
    size_t next;

    typedef std::map<T, size_t> slow_map;
    struct {
        T key{};
        size_t idx;
    } fast[FAST_MAXN];
    slow_map *slow;

    explicit Node(T key = {}) : key(key), count(), parent(0), next(1), fast(), slow(nullptr) {}
    explicit Node(T key, size_t parent, size_t next)
            : key(key), count(), parent(parent), next(next), fast(), slow(nullptr) {}
    ~Node() {}

    size_t LookupChild(T key) {
        for (auto &x: fast) {
            if (x.key == key) {
                return x.idx;
            } else if (x.key == T()) {
                return 0;
            }
        }
        if (slow == NULL) {
            return 0;
        } else {
            auto it = slow->find(key);
            if (it == slow->end()) {
                return 0;
            } else {
                return it->second;
            }
        }
    }

    void InsertChild(const T &key, size_t idx) {
        for (auto &x: fast) {
            assert(!(x.key == key));
            if (x.key == T()) {
                x.key = key;
                x.idx = idx;
                return;
            }
        }
        if (slow == NULL) {
            slow = new slow_map();
        }
        slow->emplace(key, idx);
    }
};

template<class T>
struct BbTrace {
    static constexpr int INVALID_IDX = 0;

    int _order;
    Node<T> _root;
    std::vector<std::vector<Node<T>>> _bufs;
    size_t _cur_idx;

    BbTrace(int MaxDepth) : _order(MaxDepth), _bufs(MaxDepth) {
        for (auto &buf: _bufs) {
            buf.resize(1);
        }
    }

    BbTrace(T _keys[]) {}

    Node<T> &GetNode(size_t depth, size_t idx) {
        return _bufs[depth].at(idx);
    }

    void Init(int depth, const T &key) {
        _cur_idx = WalkAtDepth(depth, 0, key);
        Count(depth, _cur_idx);
    }

    void Walk(const T &key) {
        _cur_idx = WalkAtDepth(_order - 1, _cur_idx, key);
        Count(_order - 1, _cur_idx);
    }

    /**
     * Count all ancestors of given node at @idx of level @Depth
     * @tparam Depth
     * @param idx
     */
    void Count(int depth, size_t idx) {
        Node<T> &node = GetNode(depth, idx);;
        node.count.Incement();
    }

    void Propagate() {
        this->ForeachNonZero(_order - 1, [this](Node<T> &node) {
            int next_idx = node.next;
            int addend = node.count.cur;
            for (int order = _order - 2; order >= 0; --order) {
                auto &next = GetNode(order, next_idx);
                next.count.Add(addend);
                next_idx = next.next;
            }
        });
    }


    /**
     * Insert a Node(@key, @parent) at depth @Depth and return its index in vector
     */
    size_t Insert(int depth, const T key, size_t parent, size_t next) {
        size_t idx = _bufs[depth].size();
        Node<T> node{key, parent, next};
        _bufs[depth].emplace_back(node);
        return idx;
    }

    /**
     * Walk from @idx at level @MaxDepth - 1, to next node of @key
     */
    size_t WalkAtDepth(int depth, size_t idx, const T &key) {
        if (depth == 0) {
            auto next_idx = 0;
            auto walk_idx = _root.LookupChild(key);
            if (walk_idx == INVALID_IDX) {
                size_t new_parent = next_idx;
                size_t new_next = 0;
                walk_idx = Insert(depth, key, new_parent, new_next);
                _root.InsertChild(key, walk_idx);
            }
            return walk_idx;
        } else {
            auto next_idx = GetNode(depth, idx).next;
            auto walk_idx = GetNode(depth - 1, next_idx).LookupChild(key);
            if (walk_idx == INVALID_IDX) {
                size_t new_parent = next_idx;
                size_t new_next = WalkAtDepth(depth - 1, new_parent, key);
                walk_idx = Insert(depth, key, new_parent, new_next);
                GetNode(depth - 1, next_idx).InsertChild(key, walk_idx);
            }
            return walk_idx;
        }
    }

    size_t Size(int depth) {
        return _bufs[depth].size() - 1;
    }

    template <typename F>
    void Foreach(int depth, F&& func) {
        auto &buf = _bufs[depth];
        for (size_t i = 1; i < buf.size(); ++i) {
            func(buf[i]);
        }
    }
    template <typename F>
    void ForeachNonZero(int depth, F&& func) {
        auto &buf = _bufs[depth];
        for (size_t i = 1; i < buf.size(); ++i) {
            if (buf[i].count.cur) {
                func(buf[i]);
            }
        }
    }

    void DumpDepth(int depth, std::ostream &ofs) {
        auto &buf = _bufs[depth];
        for (auto &node: buf) {
            DumpNode(depth, node, ofs);
        }
    }

    void DumpNode(int depth, Node<T> &node, std::ostream &ofs) {
        DumpNodeTrace(depth, node, ofs);
        ofs << ":" << node.count.Total() << std::endl;
    }

    void DumpNodeTrace(int depth, Node<T> &node, std::ostream &ofs) {
        if (depth > 0) {
            auto parent = GetNode(depth - 1, node.parent);
            DumpNodeTrace(depth - 1, parent, ofs);
            ofs << "-";
        }
        ofs << node.key;
    }

    template <typename F>
    void Reduce(int depth , Node<T> &node, F&& func) {
        if (depth > 0) {
            auto parent = GetNode(depth - 1, node.parent);
            Reduce(depth - 1, parent, func);
        }
        func(node);
    }

    void Dump(std::ostream &ofs) {
        ofs << std::string(64, '-') << std::endl;
        for (int i = 0; i < _order; ++i) {
            DumpDepth(i, ofs);
        }
    }
};
