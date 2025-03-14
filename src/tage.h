#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <bitset>
#include <stdint.h>
#include <cmath>
#include <vector>
#include "pad.h"

// size definitions
#define T0_COUNTER_MAX 7
#define TI_COUNTER_MAX 7
#define USE_ALT_COUNTER_MAX 7

#define BASE_PREDICTOR_SIZE 128
#define TAGE_TAG_SIZE 14
#define MAX_ALLOCATIONS 1

#define HIST_LENGTH_1 8
#define HIST_LENGTH_2 16
#define HIST_LENGTH_3 24
#define HIST_LENGTH_4 32
#define HIST_LENGTH_5 40
#define HIST_LENGTH_6 48
#define HIST_LENGTH_7 56
#define HIST_LENGTH_8 64
#define HIST_BUFFER_SIZE 64

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

class BranchPredictorTage{
private:
  std::bitset<HIST_BUFFER_SIZE> historyBuffer;
  int providerIndex;
  int altProviderIndex;
  uint32_t useAltOnNa;
  uint32_t providerPredIndex;
  bool providerPred;
  bool altProviderPred;
  
	const int table_num;
	const int index_size;

  uint32_t hist_lengths[9] = {0, /* place holder*/
    HIST_LENGTH_1, HIST_LENGTH_2, HIST_LENGTH_3, HIST_LENGTH_4,
    HIST_LENGTH_5, HIST_LENGTH_6, HIST_LENGTH_7, HIST_LENGTH_8
  };
  uint64_t tag_mask = (1ULL << TAGE_TAG_SIZE) - 1;
  uint64_t idx_mask;

  // TAGE tables
  typedef struct tageEntry{
	  uint32_t counter;
	  uint32_t tag;
	  uint8_t  useful;
  } TageEntry;
  uint32_t basePredictor[BASE_PREDICTOR_SIZE];
  std::vector< std::vector<TageEntry> > tage;
  
 public:
  // The interface to the four functions below CAN NOT be changed

  BranchPredictorTage(uint8_t _table_num, uint8_t _index_size);

  bool predict(uint64_t branchPc, bool taken, uint64_t branch_target);
  bool GetPrediction(uint64_t PC);  
  void UpdatePredictor(uint64_t PC, bool resolveDir, bool predDir, uint64_t branchTarget);
  void TrackOtherInst(uint64_t PC, uint8_t opType, uint64_t branchTarget);

  // Contestants can define their own functions below
  void GetTagePredictions(uint64_t PC, bool* usefulBitNull);
  void UpdateProviderCounter(bool resolveDir);
  void AllocateNewEntries(uint64_t PC);
  void SetU(bool truthValue);
  void UpdateHistory(bool resolveDir);
  uint64_t GetTageIndex(uint64_t PC, uint32_t tableNum);
  uint64_t GetTageTag(uint64_t PC, uint32_t tableNum);
} ATTR_LINE_ALIGNED;  // Take up an int number of cache lines

/***********************************************************/
#endif