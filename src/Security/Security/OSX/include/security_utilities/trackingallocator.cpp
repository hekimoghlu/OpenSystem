/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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
#include <security_utilities/trackingallocator.h>


//
// The default action of the destructor is to free all memory.
//
TrackingAllocator::~TrackingAllocator()
{
	reset();
}


//
// Standard allocation operations.
// We pass them down to our subAllocator and keep track of what we've got.
//
void *TrackingAllocator::malloc(size_t inSize)
{
	void *anAddress = subAllocator.malloc(inSize);
	mAllocSet.insert(anAddress);
	return anAddress;
}

void TrackingAllocator::free(void *inAddress) _NOEXCEPT
{
	subAllocator.free(inAddress);
	mAllocSet.erase(inAddress);
}

void *TrackingAllocator::realloc(void *inAddress, size_t inNewSize)
{
	void *anAddress = subAllocator.realloc(inAddress, inNewSize);
	if (anAddress != inAddress)
	{
		mAllocSet.erase(inAddress);
		mAllocSet.insert(anAddress);
	}

	return anAddress;
}


//
// Free all memory allocated through this Allocator (since the last commit(), if any)
//
void TrackingAllocator::reset()
{
	AllocSet::iterator first = mAllocSet.begin(), last = mAllocSet.end();
	for (; first != last; ++first)
		subAllocator.free(*first);
}


//
// Forget about all allocated memory. It's now your responsibility.
//
void TrackingAllocator::commit()
{
	mAllocSet.clear();
}
