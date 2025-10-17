/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
//
// cssmalloc - memory allocation in the CDSA world
//
#ifndef _H_CSSMALLOC
#define _H_CSSMALLOC

#include <security_utilities/alloc.h>
#include <Security/cssm.h>
#include <cstring>


namespace Security
{


//
// A POD wrapper for the memory functions structure passed around in CSSM.
//
class CssmMemoryFunctions : public PodWrapper<CssmMemoryFunctions, CSSM_MEMORY_FUNCS> {
public:
	CssmMemoryFunctions(const CSSM_MEMORY_FUNCS &funcs)
	{ *(CSSM_MEMORY_FUNCS *)this = funcs; }
	CssmMemoryFunctions() { }

	void *malloc(size_t size) const;
	void free(void *mem) const _NOEXCEPT { free_func(mem, AllocRef); }
	void *realloc(void *mem, size_t size) const;
	void *calloc(uint32 count, size_t size) const;
	
	bool operator == (const CSSM_MEMORY_FUNCS &other) const _NOEXCEPT
	{ return !memcmp(this, &other, sizeof(*this)); }
};

inline void *CssmMemoryFunctions::malloc(size_t size) const
{
	if (void *addr = malloc_func(size, AllocRef))
		return addr;
	throw std::bad_alloc();
}

inline void *CssmMemoryFunctions::calloc(uint32 count, size_t size) const
{
	if (void *addr = calloc_func(count, size, AllocRef))
		return addr;
	throw std::bad_alloc();
}

inline void *CssmMemoryFunctions::realloc(void *mem, size_t size) const
{
	if (void *addr = realloc_func(mem, size, AllocRef))
		return addr;
	throw std::bad_alloc();
}


//
// A Allocator based on CssmMemoryFunctions
//
class CssmMemoryFunctionsAllocator : public Allocator {
public:
	CssmMemoryFunctionsAllocator(const CssmMemoryFunctions &memFuncs) : functions(memFuncs) { }
	
	void *malloc(size_t size);
	void free(void *addr) _NOEXCEPT;
	void *realloc(void *addr, size_t size);
	
	operator const CssmMemoryFunctions & () const _NOEXCEPT { return functions; }

private:
	const CssmMemoryFunctions functions;
};


//
// A MemoryFunctions object based on a Allocator.
// Note that we don't copy the Allocator object. It needs to live (at least)
// as long as any CssmAllocatorMemoryFunctions object based on it.
//
class CssmAllocatorMemoryFunctions : public CssmMemoryFunctions {
public:
	CssmAllocatorMemoryFunctions(Allocator &alloc);
	CssmAllocatorMemoryFunctions() { /*IFDEBUG(*/ AllocRef = NULL /*)*/ ; }	// later assignment req'd
	
private:
	static void *relayMalloc(size_t size, void *ref);
	static void relayFree(void *mem, void *ref) _NOEXCEPT;
	static void *relayRealloc(void *mem, size_t size, void *ref);
	static void *relayCalloc(uint32 count, size_t size, void *ref);

	static Allocator &allocator(void *ref) _NOEXCEPT
	{ return *reinterpret_cast<Allocator *>(ref); }
};


//
// A generic helper for the unhappily ubiquitous CSSM-style
// (count, pointer-to-array) style of arrays.
//
template <class Base, class Wrapper = Base>
class CssmVector {
public:
    CssmVector(uint32 &cnt, Base * &vec, Allocator &alloc = Allocator::standard())
        : count(cnt), vector(reinterpret_cast<Wrapper * &>(vec)),
          allocator(alloc)
    {
        count = 0;
        vector = NULL;
    }
    
    ~CssmVector()	{ allocator.free(vector); }
        
    uint32 &count;
    Wrapper * &vector;
    Allocator &allocator;

public:
    Wrapper &operator [] (uint32 ix)
    { assert(ix < count); return vector[ix]; }
    
    void operator += (const Wrapper &add)
    {
        vector = reinterpret_cast<Wrapper *>(allocator.realloc(vector, (count + 1) * sizeof(Wrapper)));
        //@@@???compiler bug??? vector = allocator.alloc<Wrapper>(vector, count + 1);
        vector[count++] = add;
    }
};


} // end namespace Security

#endif //_H_CSSMALLOC
