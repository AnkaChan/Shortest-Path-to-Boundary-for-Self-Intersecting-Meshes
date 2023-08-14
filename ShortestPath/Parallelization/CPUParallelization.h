#pragma once
#include <functional>

//#define TURN_ON_DEBUG 

#if !(defined(_DEBUG) || defined(TURN_ON_DEBUG))
#define TBB_PARALLEL
#ifdef TBB_PARALLEL
#include "oneapi/tbb.h"
#endif // TBB_PARALLEL
#endif // _DEBUG
//


template<typename Func>
inline void cpu_parallel_for(int start, int end, Func & func) {
	#ifdef TBB_PARALLEL
	//std::cout << "Run in parallel on cpu.\n";
	tbb::parallel_for((int)start, (int)end, func);
	#else
	//std::cout << "Run in serial on cpu.\n";
	for (int index = start; index < end; ++index)
		func (index);
	#endif
}