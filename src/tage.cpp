#include "tage.h"
#include "mybitset.h"
#include <cassert>

/////////////// STORAGE BUDGET JUSTIFICATION ////////////////
// Total storage budget: 524288 bits
// Total size of base predictor: 48K bits
// Total size of each TAGE table group: (1 + 3 + tag bits) * entries
//		T1: (1 + 3 + 8) * 5K = 48K bits
//		T2: (1 + 3 + 11) * 16K = 240K bits
//		T3: (1 + 3 + 13) * 8K = 136K bits
//		T4: (1 + 3 + 14) * 2K = 36K bits
// Total size of TAGE predictor: 508K = 520192 bits
// History buffer size: 1728 bits
// Total control logic size: providerIndex size +
// 		altProviderIndex size + useAltOnNa size +
//		providerPredIndex size + providerPred size
//		+ altProviderPred size = 2 + 2 + 3 + 13 + 1 + 1 = 22
// Total Size: TAGE predictor size + control logic size +  
// buffer size = 485376 + 22 + 1728 = 521942 bits
/////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

BranchPredictorTage::BranchPredictorTage(uint8_t _table_num, uint8_t _index_size) : table_num(_table_num), index_size(_index_size), idx_mask((1ULL << _index_size) - 1) {
	assert(table_num <= 8);
	// Initialize control logic.
    historyBuffer = std::bitset<HIST_BUFFER_SIZE>();
    useAltOnNa = 0;
    providerPred = false;
    altProviderPred = false;
    providerPredIndex = 99999;
   
    // Initialize base predictor.
	for (uint32_t ind = 0; ind < BASE_PREDICTOR_SIZE; ind++){
	  basePredictor[ind] = T0_COUNTER_MAX / 2; 
    }

	// Initialize TAGE predictors.
	int tage_entry_count = 1 << _index_size;
	tage.resize(table_num+1);
	for(int i = 1; i <= table_num; i++){
		tage[i].resize(tage_entry_count);
		for(int j = 0; j < tage_entry_count; j++){
			tage[i][j].counter = 0;
			tage[i][j].tag = 0;
			tage[i][j].useful = false;
		}
	}
	
   providerIndex = -1, altProviderIndex = -1;
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

bool BranchPredictorTage::predict(uint64_t branchPc, bool taken, uint64_t branch_target) {
    // predict
    bool pred = GetPrediction(branchPc);
    // update
    UpdatePredictor(branchPc, taken, pred, branch_target);

    return (taken == pred);
}

bool BranchPredictorTage::GetPrediction(uint64_t PC){
	// Fetch the alternate and provider predictions from TAGE.
	// If the provider's prediction is unreliable, use the alt prediction.
	
	bool usefulBitNull = false;
	GetTagePredictions(PC, &usefulBitNull);
	if (usefulBitNull && useAltOnNa > USE_ALT_COUNTER_MAX / 2){
		return altProviderPred;
	}
	else{
		return providerPred;
	}
}


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void  BranchPredictorTage::UpdatePredictor(uint64_t PC, bool resolveDir, bool predDir, uint64_t branchTarget){
	 // Update the counter based on the branch outcome.
	 
	 UpdateProviderCounter(resolveDir);
	 
	 // In the case of a misprediction, allocate new entries on components with more history.
	 
	 if(resolveDir != predDir){
		AllocateNewEntries(PC);
	 }
	 
	 // Change the useful bit if the alt and provider predictions
	 // are different. Set based on the provider prediction accuracy.
	 // Update whether to use the alternate when the useful bit based on whether
	 // the case has occurred where the alternate prediction is correct and the
	 // provider prediction is not.
	 
	 if(altProviderPred != providerPred){
		if(altProviderPred == resolveDir){
			SetU(false);
			if(useAltOnNa < USE_ALT_COUNTER_MAX){
				 useAltOnNa++;
			}
		}
		else{
			SetU(true);
			 if(useAltOnNa > 0){
				 useAltOnNa--;
			 }
		 }
	 }
	 
	 // Update history.
	 
	 UpdateHistory(resolveDir);
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void BranchPredictorTage::TrackOtherInst(uint64_t PC, uint8_t opType, uint64_t branchTarget){
  // This function is called for instructions which are not
  // conditional branches, just in case someone decides to design
  // a predictor that uses information from such instructions.
  // We expect most contestants to leave this function untouched.

  return;
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void BranchPredictorTage::GetTagePredictions(uint64_t PC, bool* usefulBitNull){
	// Find the provider and alternate provider predictions.
	// The provider is the hitting component with the
	// longest history, and the alternate provider is the
	// hitting component with the second longest history.

	uint32_t counterVal = 0;
	providerIndex = -1;
	altProviderIndex = -1;
	providerPredIndex = 999999;
	
	for(int table=table_num; table>=1; table--){
		if(providerIndex != -1 && altProviderIndex != -1){
			break;
		}
		uint32_t idx = GetTageIndex(PC, table);
		uint32_t tag = GetTageTag(PC, table);
		if(tage[table][idx].tag == tag){
			counterVal = tage[table][idx].counter;
			if(providerIndex == -1){
				providerIndex = table;
				providerPredIndex = idx;
				if (counterVal > (TI_COUNTER_MAX / 2)){
					providerPred = true;
				}
				else{
					providerPred = false;
				}
				*usefulBitNull = tage[table][idx].useful==false;
			}
			else{
				altProviderIndex = table;
				if (counterVal >= (TI_COUNTER_MAX / 2)){
					altProviderPred = true;
				}
				else{
					altProviderPred = false;
				}
			}
		}
	}
	
	if(providerIndex == -1 || altProviderIndex == -1){
		uint32_t basePredictorIndex = PC % BASE_PREDICTOR_SIZE;
		counterVal = basePredictor[basePredictorIndex];
		if(providerIndex == -1){
			providerIndex = 0;
			providerPredIndex = basePredictorIndex;
			if (counterVal  > (T0_COUNTER_MAX / 2)){
				providerPred = true;
			}
			else{
				providerPred = false;
			}
		}
		if (altProviderIndex == -1){
			altProviderIndex = 0;
			if (counterVal > (T0_COUNTER_MAX / 2)){
				altProviderPred = true;
			}
			else{
				altProviderPred = false;
			}
		}
	}
}

void BranchPredictorTage::UpdateProviderCounter(bool resolveDir){	
	// Get the counter to update by hashing to the predictor index.
	// If the branch was not taken, decrement the counter. Otherwise, increment it.

	uint32_t* counter;
	if(providerIndex==0){
		counter = &basePredictor[providerPredIndex];
	}
	else{
		counter = &tage[providerIndex][providerPredIndex].counter;
	}
	if (resolveDir == false && (*counter) > 0){
		(*counter)--;
	}
	else if(resolveDir == true && (*counter) < T0_COUNTER_MAX){
		(*counter)++;
	}
}
	
void BranchPredictorTage::AllocateNewEntries(uint64_t PC){
	// Update the history.
	
	// Allocate up to four new entries on tables with
	// higher history tags.
	// For each entry, use the approapriate history length as the tag,
	// set the counter to weak, and set the useful bit to null.
	
	uint32_t allocationCount = 0;
	uint32_t predictorIndex;
	TageEntry newEntries[MAX_ALLOCATIONS];
	for(uint32_t i = 0; i <= MAX_ALLOCATIONS - 1; i++){
		newEntries[i].counter = TI_COUNTER_MAX / 2;
		newEntries[i].useful = false;
	}
	for(uint8_t table=providerIndex+1; table<=table_num; table++){
		if(allocationCount >= MAX_ALLOCATIONS){
			break;
		}
		predictorIndex = GetTageIndex(PC, table);
		if(tage[table][predictorIndex].useful == false){
			newEntries[allocationCount].tag = GetTageTag(PC, table);
			tage[table][predictorIndex] = newEntries[allocationCount];
			allocationCount++;
		}
		else{
			tage[table][predictorIndex].useful = false;
		}
	}

}

void BranchPredictorTage::SetU(bool truthValue){
	// Set the useful bit on the provider index.
	if(providerIndex > 0){
		tage[providerIndex][providerPredIndex].useful = truthValue;
	}
}

void BranchPredictorTage::UpdateHistory(bool resolveDir){
	 // Update history bit.
	 historyBuffer <<= 1;
	 historyBuffer[0] = resolveDir;
}

uint64_t BranchPredictorTage::GetTageIndex(uint64_t PC, uint32_t tableNum){
	if(tableNum == 0){
		return 999999;
	}
	// XOR sub-bitsets of the global history with the PC and each 
	// other to produce an index.
	
	assert(index_size <= 64);
	mybitset subHistory(index_size);
	uint32_t sm, lg;
	
	// Fold history by tag size.
	for(sm = 0, lg = hist_lengths[tableNum] - 1; sm <= lg; sm++, lg--){
		bool lastVal = subHistory.get(index_size - 1);
		bool midVal = subHistory.get((index_size / 2) - 1);
		subHistory <<= 1;
		subHistory.set(0, lastVal ^ historyBuffer[sm]);
		subHistory.set(index_size / 2, midVal ^ historyBuffer[lg]);
	}
	
	// Perform XOR with PC.
	return subHistory.to_ulong() ^ (PC & idx_mask);
}

uint64_t BranchPredictorTage::GetTageTag(uint64_t PC, uint32_t tableNum){
	if(tableNum == 0)	{
		return 999999;
	}
	// XOR sub-bitsets of the global history with the PC and each 
	// other to produce an index.
	std::bitset<TAGE_TAG_SIZE> subHistory = std::bitset<TAGE_TAG_SIZE>();
	uint32_t sm, lg;
	
	// Fold history by tag size.
	for(sm = 0, lg = hist_lengths[tableNum] - 1; sm <= lg; sm++, lg--){
		bool lastVal = subHistory[TAGE_TAG_SIZE - 1];
		bool midVal = subHistory[(TAGE_TAG_SIZE / 2) - 1];
		subHistory <<= 1;
		subHistory[0] = lastVal ^ historyBuffer[sm];
		subHistory[TAGE_TAG_SIZE / 2] = midVal ^ historyBuffer[lg];
	}
	
	// Perform XOR with PC.
	return (uint64_t)subHistory.to_ulong() ^ (PC & tag_mask);
}