/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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
#pragma once

#include <sys/cdefs.h>
#include <sys/types.h>
#include <stdbool.h>

/*
 * C: Do it ourselves.
 * Note that the runtime representation defined here should be compatible
 * with the C++ one, i.e. an _Atomic(T) needs to contain the same
 * bits as a T.
 */

#include <stddef.h>  /* For ptrdiff_t. */
#include <stdint.h>
// Include uchar.h only when available.  Bionic's stdatomic.h is also used for
// the host (via a copy in prebuilts/clang) and uchar.h is not available in the
// glibc used for the host.
#if defined(__BIONIC__)
# include <uchar.h>  /* For char16_t and char32_t.              */
#endif

/*
 * 7.17.1 Atomic lock-free macros.
 */

#define	ATOMIC_BOOL_LOCK_FREE			__CLANG_ATOMIC_BOOL_LOCK_FREE
#define	ATOMIC_CHAR_LOCK_FREE			__CLANG_ATOMIC_CHAR_LOCK_FREE
#define	ATOMIC_CHAR16_T_LOCK_FREE	__CLANG_ATOMIC_CHAR16_T_LOCK_FREE
#define	ATOMIC_CHAR32_T_LOCK_FREE	__CLANG_ATOMIC_CHAR32_T_LOCK_FREE
#define	ATOMIC_WCHAR_T_LOCK_FREE	__CLANG_ATOMIC_WCHAR_T_LOCK_FREE
#define	ATOMIC_SHORT_LOCK_FREE		__CLANG_ATOMIC_SHORT_LOCK_FREE
#define	ATOMIC_INT_LOCK_FREE			__CLANG_ATOMIC_INT_LOCK_FREE
#define	ATOMIC_LONG_LOCK_FREE			__CLANG_ATOMIC_LONG_LOCK_FREE
#define	ATOMIC_LLONG_LOCK_FREE		__CLANG_ATOMIC_LLONG_LOCK_FREE
#define	ATOMIC_POINTER_LOCK_FREE	__CLANG_ATOMIC_POINTER_LOCK_FREE

/*
 * 7.17.2 Initialization.
 */

#define	ATOMIC_VAR_INIT(value)		(value)
#define	atomic_init(obj, value)		__c11_atomic_init(obj, value)

/*
 * 7.17.3 Order and consistency.
 *
 * The memory_order_* constants that denote the barrier behaviour of the
 * atomic operations.
 * The enum values must be identical to those used by the
 * C++ <atomic> header.
 */

typedef enum {
	memory_order_relaxed = __ATOMIC_RELAXED,
	memory_order_consume = __ATOMIC_CONSUME,
	memory_order_acquire = __ATOMIC_ACQUIRE,
	memory_order_release = __ATOMIC_RELEASE,
	memory_order_acq_rel = __ATOMIC_ACQ_REL,
	memory_order_seq_cst = __ATOMIC_SEQ_CST
} memory_order;

#define kill_dependency(y) (y)

/*
 * 7.17.4 Fences.
 */

static __inline void atomic_thread_fence(memory_order __order) {
	__c11_atomic_thread_fence(__order);
}

static __inline void atomic_signal_fence(memory_order __order) {
	__c11_atomic_signal_fence(__order);
}

/*
 * 7.17.5 Lock-free property.
 */

#define	atomic_is_lock_free(obj) __c11_atomic_is_lock_free(sizeof(*(obj)))

/*
 * 7.17.6 Atomic integer types.
 */

typedef _Atomic(bool)			atomic_bool;
typedef _Atomic(char)			atomic_char;
typedef _Atomic(signed char)		atomic_schar;
typedef _Atomic(unsigned char)		atomic_uchar;
typedef _Atomic(short)			atomic_short;
typedef _Atomic(unsigned short)		atomic_ushort;
typedef _Atomic(int)			atomic_int;
typedef _Atomic(unsigned int)		atomic_uint;
typedef _Atomic(long)			atomic_long;
typedef _Atomic(unsigned long)		atomic_ulong;
typedef _Atomic(long long)		atomic_llong;
typedef _Atomic(unsigned long long)	atomic_ullong;
#if defined(__BIONIC__) || (defined(__cplusplus) && __cplusplus >= 201103L)
  typedef _Atomic(char16_t)		atomic_char16_t;
  typedef _Atomic(char32_t)		atomic_char32_t;
#endif
typedef _Atomic(wchar_t)		atomic_wchar_t;
typedef _Atomic(int_least8_t)		atomic_int_least8_t;
typedef _Atomic(uint_least8_t)	atomic_uint_least8_t;
typedef _Atomic(int_least16_t)	atomic_int_least16_t;
typedef _Atomic(uint_least16_t)	atomic_uint_least16_t;
typedef _Atomic(int_least32_t)	atomic_int_least32_t;
typedef _Atomic(uint_least32_t)	atomic_uint_least32_t;
typedef _Atomic(int_least64_t)	atomic_int_least64_t;
typedef _Atomic(uint_least64_t)	atomic_uint_least64_t;
typedef _Atomic(int_fast8_t)		atomic_int_fast8_t;
typedef _Atomic(uint_fast8_t)		atomic_uint_fast8_t;
typedef _Atomic(int_fast16_t)		atomic_int_fast16_t;
typedef _Atomic(uint_fast16_t)	atomic_uint_fast16_t;
typedef _Atomic(int_fast32_t)		atomic_int_fast32_t;
typedef _Atomic(uint_fast32_t)	atomic_uint_fast32_t;
typedef _Atomic(int_fast64_t)		atomic_int_fast64_t;
typedef _Atomic(uint_fast64_t)	atomic_uint_fast64_t;
typedef _Atomic(intptr_t)		atomic_intptr_t;
typedef _Atomic(uintptr_t)		atomic_uintptr_t;
typedef _Atomic(size_t)		atomic_size_t;
typedef _Atomic(ptrdiff_t)		atomic_ptrdiff_t;
typedef _Atomic(intmax_t)		atomic_intmax_t;
typedef _Atomic(uintmax_t)		atomic_uintmax_t;

/*
 * 7.17.7 Operations on atomic types.
 */

/*
 * Compiler-specific operations.
 */

#define	atomic_compare_exchange_strong_explicit(object, expected,	\
    desired, success, failure)						\
	__c11_atomic_compare_exchange_strong(object, expected, desired,	\
	    success, failure)
#define	atomic_compare_exchange_weak_explicit(object, expected,		\
    desired, success, failure)						\
	__c11_atomic_compare_exchange_weak(object, expected, desired,	\
	    success, failure)
#define	atomic_exchange_explicit(object, desired, order)		\
	__c11_atomic_exchange(object, desired, order)
#define	atomic_fetch_add_explicit(object, operand, order)		\
	__c11_atomic_fetch_add(object, operand, order)
#define	atomic_fetch_and_explicit(object, operand, order)		\
	__c11_atomic_fetch_and(object, operand, order)
#define	atomic_fetch_or_explicit(object, operand, order)		\
	__c11_atomic_fetch_or(object, operand, order)
#define	atomic_fetch_sub_explicit(object, operand, order)		\
	__c11_atomic_fetch_sub(object, operand, order)
#define	atomic_fetch_xor_explicit(object, operand, order)		\
	__c11_atomic_fetch_xor(object, operand, order)
#define	atomic_load_explicit(object, order)				\
	__c11_atomic_load(object, order)
#define	atomic_store_explicit(object, desired, order)			\
	__c11_atomic_store(object, desired, order)

/*
 * Convenience functions.
 */

#define	atomic_compare_exchange_strong(object, expected, desired)	\
	atomic_compare_exchange_strong_explicit(object, expected,	\
	    desired, memory_order_seq_cst, memory_order_seq_cst)
#define	atomic_compare_exchange_weak(object, expected, desired)		\
	atomic_compare_exchange_weak_explicit(object, expected,		\
	    desired, memory_order_seq_cst, memory_order_seq_cst)
#define	atomic_exchange(object, desired)				\
	atomic_exchange_explicit(object, desired, memory_order_seq_cst)
#define	atomic_fetch_add(object, operand)				\
	atomic_fetch_add_explicit(object, operand, memory_order_seq_cst)
#define	atomic_fetch_and(object, operand)				\
	atomic_fetch_and_explicit(object, operand, memory_order_seq_cst)
#define	atomic_fetch_or(object, operand)				\
	atomic_fetch_or_explicit(object, operand, memory_order_seq_cst)
#define	atomic_fetch_sub(object, operand)				\
	atomic_fetch_sub_explicit(object, operand, memory_order_seq_cst)
#define	atomic_fetch_xor(object, operand)				\
	atomic_fetch_xor_explicit(object, operand, memory_order_seq_cst)
#define	atomic_load(object)						\
	atomic_load_explicit(object, memory_order_seq_cst)
#define	atomic_store(object, desired)					\
	atomic_store_explicit(object, desired, memory_order_seq_cst)

/*
 * 7.17.8 Atomic flag type and operations.
 *
 * atomic_bool can be used to provide a lock-free atomic flag type on every
 * Android architecture, so this shouldn't be needed in new Android code,
 * but is in ISO C, and available for portability to PA-RISC and
 * microcontrollers.
 */

typedef struct {
	atomic_bool	__flag;
} atomic_flag;

#define	ATOMIC_FLAG_INIT {false}

static __inline bool atomic_flag_test_and_set_explicit(volatile atomic_flag * _Nonnull __object, memory_order __order) {
	return (atomic_exchange_explicit(&__object->__flag, 1, __order));
}

static __inline void atomic_flag_clear_explicit(volatile atomic_flag * _Nonnull __object, memory_order __order) {
	atomic_store_explicit(&__object->__flag, 0, __order);
}

static __inline bool atomic_flag_test_and_set(volatile atomic_flag * _Nonnull __object) {
	return (atomic_flag_test_and_set_explicit(__object, memory_order_seq_cst));
}

static __inline void atomic_flag_clear(volatile atomic_flag * _Nonnull __object) {
	atomic_flag_clear_explicit(__object, memory_order_seq_cst);
}
