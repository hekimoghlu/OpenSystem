/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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
// cssmalloc - memory allocation in the CDSA world.
//
// Don't eat heavily before inspecting this code.
//
#define __STDC_WANT_LIB_EXT1__ 1
#include <string.h>

#include <security_utilities/alloc.h>
#include <security_utilities/memutils.h>
#include <security_utilities/globalizer.h>
#include <stdlib.h>
#include <errno.h>

using LowLevelMemoryUtilities::alignof_template;
using LowLevelMemoryUtilities::increment;
using LowLevelMemoryUtilities::alignUp;

extern "C" size_t malloc_size(void *);


//
// Features of the Allocator root class
//
bool Allocator::operator == (const Allocator &alloc) const _NOEXCEPT
{
	return this == &alloc;
}

Allocator::~Allocator()
{
}


//
// Standard Allocator variants.
// Note that all calls to Allocator::standard(xxx) with the same xxx argument
// must produce compatible allocators (i.e. they must be work on a common memory
// pool). This is trivially achieved here by using singletons.
//
struct DefaultAllocator : public Allocator {
	void *malloc(size_t size);
	void free(void *addr) _NOEXCEPT;
	void *realloc(void *addr, size_t size);
};

struct SensitiveAllocator : public DefaultAllocator {
    void free(void *addr) _NOEXCEPT;
    void *realloc(void *addr, size_t size);
};

struct DefaultAllocators {
    DefaultAllocator standard;
    SensitiveAllocator sensitive;
};

static ModuleNexus<DefaultAllocators> defaultAllocators;


Allocator &Allocator::standard(UInt32 request)
{
    switch (request) {
    case normal:
        return defaultAllocators().standard;
    case sensitive:
        return defaultAllocators().sensitive;
    default:
        UnixError::throwMe(ENOMEM);
    }
}

void *DefaultAllocator::malloc(size_t size)
{
	if (void *result = ::malloc(size))
		return result;
	throw std::bad_alloc();
}

void DefaultAllocator::free(void *addr) _NOEXCEPT
{
	::free(addr);
}

void *DefaultAllocator::realloc(void *addr, size_t newSize)
{
	if (void *result = ::realloc(addr, newSize))
		return result;
	throw std::bad_alloc();
}

void SensitiveAllocator::free(void *addr) _NOEXCEPT
{
    size_t size = malloc_size(addr);
    ::memset_s(addr, size, 0, size);
    DefaultAllocator::free(addr);
}

void *SensitiveAllocator::realloc(void *addr, size_t newSize)
{
    size_t oldSize = malloc_size(addr);
    if (newSize < oldSize)
        ::memset_s(increment(addr, newSize), oldSize - newSize, 0, oldSize - newSize);
    return DefaultAllocator::realloc(addr, newSize);
}


//
// Memory allocators for CssmHeap objects.
// This implementation stores a pointer to the allocator used into memory
// *after* the object's proper storage block. This allows the usual free()
// functions to safely free our (hidden) pointer without knowing about it.
// An allocator argument of NULL is interpreted as the standard allocator.
//
void *CssmHeap::operator new (size_t size, Allocator *alloc)
{
    if (size > SIZE_T_MAX / 2) {
        throw std::bad_alloc();
    }
    if (alloc == NULL) {
		alloc = &Allocator::standard();
    }
	size = alignUp(size, alignof_template<Allocator *>());
	size_t totalSize = size + sizeof(Allocator *);
	void *addr = alloc->malloc(totalSize);
	*(Allocator **)increment(addr, size) = alloc;
	return addr;
}

void CssmHeap::operator delete (void *addr, size_t size, Allocator *alloc) _NOEXCEPT
{
	alloc->free(addr);	// as per C++ std, called (only) if construction fails
}

void CssmHeap::operator delete (void *addr, size_t size) _NOEXCEPT
{
	void *end = increment(addr, alignUp(size, alignof_template<Allocator *>()));
	(*(Allocator **)end)->free(addr);
}


//
// CssmVector
//
