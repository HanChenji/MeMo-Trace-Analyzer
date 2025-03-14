
#ifndef PROFILE_H
#define PROFILE_H


#include <map>
#include <unordered_set>
#include <queue>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdio.h>

#include "pin.H"
#include "zfstream.h"

typedef std::queue<UINT64> REGION_LENGTHS_QUEUE;


/* This class holds the file descriptors  of output file*/
class PROFILE
{ 
  private:
    static const uint32_t BUFSIZE = 100;

  public:
    PROFILE(int order);
    ~PROFILE();

    void OpenFile(THREADID tid, std::string output_file);

    void OpenFilePerOrder(int order, THREADID tid, std::string output_file);

    void CloseAll();

    int   _order;
    BOOL  first;

    gzofstream* BbFile;
    REGION_LENGTHS_QUEUE length_queue;
};


#endif