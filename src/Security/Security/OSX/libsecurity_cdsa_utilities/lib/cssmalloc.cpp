/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
#include <security_cdsa_utilities/cssmalloc.h>
#include <stdlib.h>
#include <errno.h>
#include <os/overflow.h>



namespace Security {


//
// CssmMemoryFunctionsAllocators
//
void *CssmMemoryFunctionsAllocator::malloc(size_t size)
{ return functions.malloc(size); }

void CssmMemoryFunctionsAllocator::free(void *addr) _NOEXCEPT
{ return functions.free(addr); }

void *CssmMemoryFunctionsAllocator::realloc(void *addr, size_t size)
{ return functions.realloc(addr, size); }


//
// CssmAllocatorMemoryFunctions
//
CssmAllocatorMemoryFunctions::CssmAllocatorMemoryFunctions(Allocator &alloc)
{
	AllocRef = &alloc;
	malloc_func = relayMalloc;
	free_func = relayFree;
	realloc_func = relayRealloc;
	calloc_func = relayCalloc;
}

void *CssmAllocatorMemoryFunctions::relayMalloc(size_t size, void *ref)
{ return allocator(ref).malloc(size); }

void CssmAllocatorMemoryFunctions::relayFree(void *mem, void *ref) _NOEXCEPT
{ allocator(ref).free(mem); }

void *CssmAllocatorMemoryFunctions::relayRealloc(void *mem, size_t size, void *ref)
{ return allocator(ref).realloc(mem, size); }

void *CssmAllocatorMemoryFunctions::relayCalloc(uint32 count, size_t size, void *ref)
{
	// Allocator doesn't have a calloc() method
	size_t alloc_size = 0;
	if (os_mul_overflow(count, size, &alloc_size)) {
		return NULL;
	}
	void *mem = allocator(ref).malloc(alloc_size);
	memset(mem, 0, alloc_size);
	return mem;
}


//
// CssmVector
//


}   // namespace Security
