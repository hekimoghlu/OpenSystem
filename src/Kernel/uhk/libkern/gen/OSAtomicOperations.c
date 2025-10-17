/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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
#include <libkern/OSAtomic.h>
#include <kern/debug.h>
#include <machine/atomic.h>
#include <stdbool.h>

#ifndef NULL
#define NULL ((void *)0)
#endif

#define ATOMIC_DEBUG DEBUG

#if ATOMIC_DEBUG
#define ALIGN_TEST(p, t) do{if((uintptr_t)p&(sizeof(t)-1)) panic("Unaligned atomic pointer %p",p);}while(0)
#else
#define ALIGN_TEST(p, t) do{}while(0)
#endif

/*
 * atomic operations
 *	These are _the_ atomic operations, now implemented via compiler built-ins.
 *	It is expected that this C implementation is a candidate for Link-Time-
 *	Optimization inlining, whereas the assembler implementations they replace
 *	were not.
 */

#undef OSCompareAndSwap8
Boolean
OSCompareAndSwap8(UInt8 oldValue, UInt8 newValue, volatile UInt8 *address)
{
	return (Boolean)os_atomic_cmpxchg(address, oldValue, newValue, acq_rel);
}

#undef OSCompareAndSwap16
Boolean
OSCompareAndSwap16(UInt16 oldValue, UInt16 newValue, volatile UInt16 *address)
{
	return (Boolean)os_atomic_cmpxchg(address, oldValue, newValue, acq_rel);
}

#undef OSCompareAndSwap
Boolean
OSCompareAndSwap(UInt32 oldValue, UInt32 newValue, volatile UInt32 *address)
{
	ALIGN_TEST(address, UInt32);
	return (Boolean)os_atomic_cmpxchg(address, oldValue, newValue, acq_rel);
}

#undef OSCompareAndSwap64
Boolean
OSCompareAndSwap64(UInt64 oldValue, UInt64 newValue, volatile UInt64 *address)
{
	/*
	 * _Atomic uint64 requires 8-byte alignment on all architectures.
	 * This silences the compiler cast warning.  ALIGN_TEST() verifies
	 * that the cast was legal, if defined.
	 */
	_Atomic UInt64 *aligned_addr = (_Atomic UInt64 *)(uintptr_t)address;

	ALIGN_TEST(address, UInt64);
	return (Boolean)os_atomic_cmpxchg(aligned_addr, oldValue, newValue, acq_rel);
}

#undef OSCompareAndSwapPtr
Boolean
OSCompareAndSwapPtr(void *oldValue, void *newValue, void * volatile *address)
{
	return (Boolean)os_atomic_cmpxchg(address, oldValue, newValue, acq_rel);
}

SInt8
OSAddAtomic8(SInt32 amount, volatile SInt8 *address)
{
	return os_atomic_add_orig(address, (SInt8)amount, relaxed);
}

SInt16
OSAddAtomic16(SInt32 amount, volatile SInt16 *address)
{
	return os_atomic_add_orig(address, (SInt16)amount, relaxed);
}

#undef OSAddAtomic
SInt32
OSAddAtomic(SInt32 amount, volatile SInt32 *address)
{
	ALIGN_TEST(address, UInt32);
	return os_atomic_add_orig(address, amount, relaxed);
}

#undef OSAddAtomic64
SInt64
OSAddAtomic64(SInt64 amount, volatile SInt64 *address)
{
	_Atomic SInt64* aligned_address = (_Atomic SInt64*)(uintptr_t)address;

	ALIGN_TEST(address, SInt64);
	return os_atomic_add_orig(aligned_address, amount, relaxed);
}

#undef OSAddAtomicLong
long
OSAddAtomicLong(long theAmount, volatile long *address)
{
	return os_atomic_add_orig(address, theAmount, relaxed);
}

#undef OSIncrementAtomic
SInt32
OSIncrementAtomic(volatile SInt32 * value)
{
	return os_atomic_inc_orig(value, relaxed);
}

#undef OSDecrementAtomic
SInt32
OSDecrementAtomic(volatile SInt32 * value)
{
	return os_atomic_dec_orig(value, relaxed);
}

#undef OSBitAndAtomic
UInt32
OSBitAndAtomic(UInt32 mask, volatile UInt32 * value)
{
	return os_atomic_and_orig(value, mask, relaxed);
}

#undef OSBitOrAtomic
UInt32
OSBitOrAtomic(UInt32 mask, volatile UInt32 * value)
{
	return os_atomic_or_orig(value, mask, relaxed);
}

#undef OSBitXorAtomic
UInt32
OSBitXorAtomic(UInt32 mask, volatile UInt32 * value)
{
	return os_atomic_xor_orig(value, mask, relaxed);
}

static Boolean
OSTestAndSetClear(UInt32 bit, bool wantSet, volatile UInt8 * startAddress)
{
	UInt8           mask = 1;
	UInt8           oldValue, newValue;
	UInt8           wantValue;
	UInt8           *address;

	address = (UInt8 *)(uintptr_t)(startAddress + (bit / 8));
	mask <<= (7 - (bit % 8));
	wantValue = wantSet ? mask : 0;

	return !os_atomic_rmw_loop(address, oldValue, newValue, relaxed, {
		if ((oldValue & mask) == wantValue) {
		        os_atomic_rmw_loop_give_up(break);
		}
		newValue = (oldValue & ~mask) | wantValue;
	});
}

Boolean
OSTestAndSet(UInt32 bit, volatile UInt8 * startAddress)
{
	return OSTestAndSetClear(bit, true, startAddress);
}

Boolean
OSTestAndClear(UInt32 bit, volatile UInt8 * startAddress)
{
	return OSTestAndSetClear(bit, false, startAddress);
}

/*
 * silly unaligned versions
 */

SInt8
OSIncrementAtomic8(volatile SInt8 * value)
{
	return os_atomic_inc_orig(value, relaxed);
}

SInt8
OSDecrementAtomic8(volatile SInt8 * value)
{
	return os_atomic_dec_orig(value, relaxed);
}

UInt8
OSBitAndAtomic8(UInt32 mask, volatile UInt8 * value)
{
	return os_atomic_and_orig(value, (UInt8)mask, relaxed);
}

UInt8
OSBitOrAtomic8(UInt32 mask, volatile UInt8 * value)
{
	return os_atomic_or_orig(value, (UInt8)mask, relaxed);
}

UInt8
OSBitXorAtomic8(UInt32 mask, volatile UInt8 * value)
{
	return os_atomic_xor_orig(value, (UInt8)mask, relaxed);
}

SInt16
OSIncrementAtomic16(volatile SInt16 * value)
{
	return OSAddAtomic16(1, value);
}

SInt16
OSDecrementAtomic16(volatile SInt16 * value)
{
	return OSAddAtomic16(-1, value);
}

UInt16
OSBitAndAtomic16(UInt32 mask, volatile UInt16 * value)
{
	return os_atomic_and_orig(value, (UInt16)mask, relaxed);
}

UInt16
OSBitOrAtomic16(UInt32 mask, volatile UInt16 * value)
{
	return os_atomic_or_orig(value, (UInt16)mask, relaxed);
}

UInt16
OSBitXorAtomic16(UInt32 mask, volatile UInt16 * value)
{
	return os_atomic_xor_orig(value, (UInt16)mask, relaxed);
}
