/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
// Tests for
//  T& operator[](std::ptrdiff_t n);
//  T const& operator[](std::ptrdiff_t n) const;
//

#include <libkern/c++/safe_allocation.h>
#include <darwintest.h>
#include "test_utils.h"
#include <cstddef>
#include <limits>

struct T {
	long i;
};

template <typename RawT, typename QualT>
static void
tests()
{
	// Test the non-const version
	{
		RawT* memory = reinterpret_cast<RawT*>(malloc_allocator::allocate(10 * sizeof(RawT)));
		for (RawT* ptr = memory; ptr != memory + 10; ++ptr) {
			*ptr = RawT{ptr - memory}; // number from 0 to 9
		}

		test_safe_allocation<QualT> array(memory, 10, libkern::adopt_memory);
		for (std::ptrdiff_t n = 0; n != 10; ++n) {
			QualT& element = array[n];
			CHECK(&element == memory + n);
		}
	}

	// Test the const version
	{
		RawT* memory = reinterpret_cast<RawT*>(malloc_allocator::allocate(10 * sizeof(RawT)));
		for (RawT* ptr = memory; ptr != memory + 10; ++ptr) {
			*ptr = RawT{ptr - memory}; // number from 0 to 9
		}

		test_safe_allocation<QualT> const array(memory, 10, libkern::adopt_memory);
		for (std::ptrdiff_t n = 0; n != 10; ++n) {
			QualT const& element = array[n];
			CHECK(&element == memory + n);
		}
	}

	// Test with OOB offsets (should trap)
	{
		using Alloc = libkern::safe_allocation<RawT, malloc_allocator, tracking_trapping_policy>;
		RawT* memory = reinterpret_cast<RawT*>(malloc_allocator::allocate(10 * sizeof(RawT)));
		Alloc const array(memory, 10, libkern::adopt_memory);

		// Negative offsets
		{
			tracking_trapping_policy::reset();
			(void)array[-1];
			CHECK(tracking_trapping_policy::did_trap);
		}
		{
			tracking_trapping_policy::reset();
			(void)array[-10];
			CHECK(tracking_trapping_policy::did_trap);
		}

		// Too large offsets
		{
			tracking_trapping_policy::reset();
			(void)array[10];
			CHECK(tracking_trapping_policy::did_trap);
		}
		{
			tracking_trapping_policy::reset();
			(void)array[11];
			CHECK(tracking_trapping_policy::did_trap);
		}
	}
}

T_DECL(operator_subscript, "safe_allocation.operator.subscript", T_META_TAG_VM_PREFERRED) {
	tests<T, T>();
	tests<T, T const>();
}
