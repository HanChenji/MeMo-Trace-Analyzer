#!/bin/bash

PROFILE() {
    local config=$1
    ./run-MeMo.py -t profiling -p cpu2017-gcc_r-4 -n 1 --config $config
}

PROFILE ooo
PROFILE BbvProfiler

PROFILE FetchModel-cachex1
PROFILE FetchModel-widthx1
PROFILE FetchModel-bpx1
PROFILE IssueModelx1
PROFILE CacheModelx1

PROFILE FetchModel-widthx2
PROFILE FetchModel-cachex2
PROFILE FetchModel-bpx2
PROFILE IssueModelx2
PROFILE CacheModelx2

PROFILE IssueModelx4
PROFILE FetchModel-x4
PROFILE CacheModelx4

PROFILE FetchModel-widthx8
PROFILE FetchModel-cachex8
PROFILE FetchModel-bpx8
PROFILE IssueModelx8
PROFILE CacheModelx8

PROFILE CacheModelx16
PROFILE IssueModelx16
PROFILE FetchModel-bpx16
PROFILE FetchModel-cachex16
PROFILE FetchModel-widthx16
