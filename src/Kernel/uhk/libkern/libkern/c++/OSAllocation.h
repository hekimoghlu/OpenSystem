/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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
#ifndef XNU_LIBKERN_LIBKERN_CXX_OS_ALLOCATION_H
#define XNU_LIBKERN_LIBKERN_CXX_OS_ALLOCATION_H

#if !TAPI

#include <stddef.h>
#if DRIVERKIT_FRAMEWORK_INCLUDE
#include <DriverKit/OSBoundedPtr.h>
#include <DriverKit/safe_allocation.h>
#include <DriverKit/IOLib.h> // IOMalloc/IOFree
#else
#include <libkern/c++/OSBoundedPtr.h>
#include <libkern/c++/safe_allocation.h>
#include <IOKit/IOLib.h> // IOMalloc/IOFree
#endif /* DRIVERKIT_FRAMEWORK_INCLUDE */

namespace os_detail {
struct IOKit_allocator {
	static void*
	allocate(size_t bytes)
	{
		return IOMalloc(bytes);
	}

	static void*
	allocate_zero(size_t bytes)
	{
		return IOMallocZero(bytes);
	}

	static void
	deallocate(void* p, size_t bytes)
	{
		IOFree(p, bytes);
	}
};

#ifdef KERNEL_PRIVATE
struct IOKit_data_allocator {
	static void*
	allocate(size_t bytes)
	{
		return IOMallocData(bytes);
	}

	static void*
	allocate_zero(size_t bytes)
	{
		return IOMallocZeroData(bytes);
	}

	static void
	deallocate(void* p, size_t bytes)
	{
		IOFreeData(p, bytes);
	}
};

template <typename T>
constexpr bool IOKit_is_data_v = KALLOC_TYPE_IS_DATA_ONLY(T);

template <typename T, bool DataOnly = IOKit_is_data_v<T> >
struct IOKit_typed_allocator;

template <typename T>
struct IOKit_typed_allocator<T, false> {
	static inline kalloc_type_var_view_t
	kt_view(void)
	{
		static KALLOC_TYPE_VAR_DEFINE(kt_view, T, KT_SHARED_ACCT);
		return kt_view;
	}

	static void*
	allocate(size_t bytes)
	{
		return IOMallocTypeVarImpl(kt_view(), bytes);
	}

	static void*
	allocate_zero(size_t bytes)
	{
		return IOMallocTypeVarImpl(kt_view(), bytes);
	}

	static void
	deallocate(void* p, size_t bytes)
	{
		IOFreeTypeVarImpl(kt_view(), p, bytes);
	}
};

template <typename T>
struct IOKit_typed_allocator<T, true> : IOKit_data_allocator {
};
#endif
} // end namespace os_detail

template <
	typename T,
#if KERNEL_PRIVATE
	typename Allocator = os_detail::IOKit_typed_allocator<T>
#else
	typename Allocator = os_detail::IOKit_allocator
#endif
	>
using OSAllocation = libkern::safe_allocation<T, Allocator, os_detail::panic_trapping_policy>;

#ifdef KERNEL_PRIVATE
// obsolete: just use the determination that OSAllocation<> does already.
//           (this works around incorrect adoptions like 104478984).
template <typename T>
using OSDataAllocation = OSAllocation<T>;
#endif

inline constexpr auto OSAllocateMemory = libkern::allocate_memory;
inline constexpr auto OSAllocateMemoryZero = libkern::allocate_memory_zero;
inline constexpr auto OSAdoptMemory = libkern::adopt_memory;

#endif /* !TAPI */

#endif /* !XNU_LIBKERN_LIBKERN_CXX_OS_ALLOCATION_H */
