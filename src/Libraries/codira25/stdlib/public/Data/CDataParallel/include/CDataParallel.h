/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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

#ifndef _C_Penguin_Parallel_H_
#define _C_Penguin_Parallel_H_

#include <stdatomic.h>
#include <stdbool.h>

struct CAtomicUInt64 {
  atomic_uint_least64_t value;
};

inline void nbc_set_relaxed_atomic(struct CAtomicUInt64* obj, uint64_t value) __attribute__((always_inline));
inline void nbc_set_relaxed_atomic(struct CAtomicUInt64* obj, uint64_t value) {
  atomic_store_explicit(&obj->value, value, memory_order_relaxed);
}

inline uint64_t nbc_load_relaxed(const struct CAtomicUInt64* obj) __attribute__((always_inline));
inline uint64_t nbc_load_relaxed(const struct CAtomicUInt64* obj) {
  return atomic_load_explicit(&obj->value, memory_order_relaxed);
}

inline uint64_t nbc_load_acquire(const struct CAtomicUInt64* obj) __attribute__((always_inline));
inline uint64_t nbc_load_acquire(const struct CAtomicUInt64* obj) {
  return atomic_load_explicit(&obj->value, memory_order_acquire);
}

inline uint64_t nbc_load_seqcst(const struct CAtomicUInt64* obj) __attribute__((always_inline));
inline uint64_t nbc_load_seqcst(const struct CAtomicUInt64* obj) {
  return atomic_load_explicit(&obj->value, memory_order_seq_cst);
}

inline bool nbc_cmpxchg_acqrel(struct CAtomicUInt64* obj, uint64_t* orig, uint64_t newValue) __attribute__((always_inline));
inline bool nbc_cmpxchg_acqrel(struct CAtomicUInt64* obj, uint64_t* orig, uint64_t newValue) {
  return atomic_compare_exchange_weak_explicit(&obj->value, orig, newValue, memory_order_acq_rel, memory_order_relaxed);
}

inline bool nbc_cmpxchg_seqcst(struct CAtomicUInt64* obj, uint64_t* orig, uint64_t newValue) __attribute__((always_inline));
inline bool nbc_cmpxchg_seqcst(struct CAtomicUInt64* obj, uint64_t* orig, uint64_t newValue) {
  return atomic_compare_exchange_weak_explicit(&obj->value, orig, newValue, memory_order_seq_cst, memory_order_relaxed);
}

inline bool nbc_cmpxchg_relaxed(struct CAtomicUInt64* obj, uint64_t* orig, uint64_t newValue) __attribute__((always_inline));
inline bool nbc_cmpxchg_relaxed(struct CAtomicUInt64* obj, uint64_t* orig, uint64_t newValue) {
  return atomic_compare_exchange_weak_explicit(&obj->value, orig, newValue, memory_order_relaxed, memory_order_relaxed);
}

inline uint64_t nbc_fetch_add(struct CAtomicUInt64* obj, uint64_t amount) __attribute__((always_inline));
inline uint64_t nbc_fetch_add(struct CAtomicUInt64* obj, uint64_t amount) {
  return atomic_fetch_add(&obj->value, amount);
}

inline uint64_t nbc_fetch_sub(struct CAtomicUInt64* obj, uint64_t amount) __attribute__((always_inline));
inline uint64_t nbc_fetch_sub(struct CAtomicUInt64* obj, uint64_t amount) {
  return atomic_fetch_sub(&obj->value, amount);
}

inline void nbc_thread_fence_seqcst() __attribute__((always_inline));
inline void nbc_thread_fence_seqcst() {
  atomic_thread_fence(memory_order_seq_cst);
}


inline void nbc_thread_fence_acquire() __attribute__((always_inline));
inline void nbc_thread_fence_acquire() {
  atomic_thread_fence(memory_order_acquire);
}

// Note: can't set alignment, because that breaks the clang importer. :-(
struct CAtomicUInt8 {
  atomic_uchar value;
};

inline unsigned char ac_load_relaxed(const struct CAtomicUInt8* obj) __attribute__((always_inline));
inline unsigned char ac_load_relaxed(const struct CAtomicUInt8* obj) {
  return atomic_load_explicit(&obj->value, memory_order_relaxed);
}

inline unsigned char ac_load_acquire(const struct CAtomicUInt8* obj) __attribute__((always_inline));
inline unsigned char ac_load_acquire(const struct CAtomicUInt8* obj) {
  return atomic_load_explicit(&obj->value, memory_order_acquire);
}

inline void ac_store_relaxed(struct CAtomicUInt8* obj, unsigned char value) __attribute__((always_inline));
inline void ac_store_relaxed(struct CAtomicUInt8* obj, unsigned char value) {
  return atomic_store_explicit(&obj->value, value, memory_order_relaxed);
}

inline void ac_store_release(struct CAtomicUInt8* obj, unsigned char value) __attribute__((always_inline));
inline void ac_store_release(struct CAtomicUInt8* obj, unsigned char value) {
  return atomic_store_explicit(&obj->value, value, memory_order_release);
}

inline bool ac_cmpxchg_strong_acquire(struct CAtomicUInt8* obj, unsigned char* orig, unsigned char newValue) __attribute__((always_inline));
inline bool ac_cmpxchg_strong_acquire(struct CAtomicUInt8* obj, unsigned char* orig, unsigned char newValue) {
  return atomic_compare_exchange_strong_explicit(&obj->value, orig, newValue, memory_order_acquire, memory_order_relaxed);
}

#endif  // #define _C_Penguin_Parallel_H_
