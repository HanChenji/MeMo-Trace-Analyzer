#include "Profile.H"
#include "core.h"

PROFILE::PROFILE(int order):_order(order), first(true)
{
    BbFile  = new gzofstream[_order+1];
}

PROFILE::~PROFILE()
{
    for(int i=0; i<=_order; i++)
    {
        if(BbFile[i].is_open())
            BbFile[i].close();
    }
    delete [] BbFile;
}

void PROFILE::CloseAll()
{
    for(int order=0; order<=_order; order++){
        if(BbFile[order].is_open())
            BbFile[order].close();
    }
}

void PROFILE::OpenFile(THREADID tid, std::string output_file)
{
    for(int order=0; order<=_order; order++)
    {
        OpenFilePerOrder(order, tid, output_file);
    } 
}

void PROFILE::OpenFilePerOrder(int order, THREADID tid, std::string output_file)
{
    gzofstream& cur_order_bbfile  = BbFile[order];
    if (!cur_order_bbfile.is_open())
    {
        std::string tname = ".T." + std::to_string((int)tid) + ".order" + std::to_string(order);
        cur_order_bbfile.open((output_file + tname + ".bb.gz").c_str());
        cur_order_bbfile.setf(std::ios::showbase);
    }
}