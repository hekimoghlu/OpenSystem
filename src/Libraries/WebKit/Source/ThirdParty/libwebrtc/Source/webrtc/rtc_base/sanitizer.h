/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
#ifndef RTC_BASE_SANITIZER_H_
#define RTC_BASE_SANITIZER_H_

#include <stddef.h>  // For size_t.

#ifdef __cplusplus
#include "absl/meta/type_traits.h"
#endif

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define RTC_HAS_ASAN 1
#endif
#if __has_feature(memory_sanitizer)
#define RTC_HAS_MSAN 1
#endif
#endif
#ifndef RTC_HAS_ASAN
#define RTC_HAS_ASAN 0
#endif
#ifndef RTC_HAS_MSAN
#define RTC_HAS_MSAN 0
#endif

#if RTC_HAS_ASAN
#include <sanitizer/asan_interface.h>
#endif
#if RTC_HAS_MSAN
#include <sanitizer/msan_interface.h>
#endif

#ifdef __has_attribute
#if __has_attribute(no_sanitize)
#define RTC_NO_SANITIZE(what) __attribute__((no_sanitize(what)))
#endif
#endif
#ifndef RTC_NO_SANITIZE
#define RTC_NO_SANITIZE(what)
#endif

// Ask ASan to mark the memory range [ptr, ptr + element_size * num_elements)
// as being unaddressable, so that reads and writes are not allowed. ASan may
// narrow the range to the nearest alignment boundaries.
static inline void rtc_AsanPoison(const volatile void* ptr,
                                  size_t element_size,
                                  size_t num_elements) {
#if RTC_HAS_ASAN
  ASAN_POISON_MEMORY_REGION(ptr, element_size * num_elements);
#else
  // This is to prevent from the compiler raising a warning/error over unused
  // variables. We cannot use clang's annotation (`[[maybe_unused]]`) because
  // this file is also included from c files which doesn't support the
  // annotation till we switch to C23
  (void)ptr;
  (void)element_size;
  (void)num_elements;
#endif
}

// Ask ASan to mark the memory range [ptr, ptr + element_size * num_elements)
// as being addressable, so that reads and writes are allowed. ASan may widen
// the range to the nearest alignment boundaries.
static inline void rtc_AsanUnpoison(const volatile void* ptr,
                                    size_t element_size,
                                    size_t num_elements) {
#if RTC_HAS_ASAN
  ASAN_UNPOISON_MEMORY_REGION(ptr, element_size * num_elements);
#else
  (void)ptr;
  (void)element_size;
  (void)num_elements;
#endif
}

// Ask MSan to mark the memory range [ptr, ptr + element_size * num_elements)
// as being uninitialized.
static inline void rtc_MsanMarkUninitialized(const volatile void* ptr,
                                             size_t element_size,
                                             size_t num_elements) {
#if RTC_HAS_MSAN
  __msan_poison(ptr, element_size * num_elements);
#else
  (void)ptr;
  (void)element_size;
  (void)num_elements;
#endif
}

// Force an MSan check (if any bits in the memory range [ptr, ptr +
// element_size * num_elements) are uninitialized the call will crash with an
// MSan report).
static inline void rtc_MsanCheckInitialized(const volatile void* ptr,
                                            size_t element_size,
                                            size_t num_elements) {
#if RTC_HAS_MSAN
  __msan_check_mem_is_initialized(ptr, element_size * num_elements);
#else
  (void)ptr;
  (void)element_size;
  (void)num_elements;
#endif
}

#ifdef __cplusplus

namespace rtc {
namespace sanitizer_impl {

template <typename T>
constexpr bool IsTriviallyCopyable() {
  return static_cast<bool>(absl::is_trivially_copy_constructible<T>::value &&
                           (absl::is_trivially_copy_assignable<T>::value ||
                            !std::is_copy_assignable<T>::value) &&
                           absl::is_trivially_destructible<T>::value);
}

}  // namespace sanitizer_impl

template <typename T>
inline void AsanPoison(const T& mem) {
  rtc_AsanPoison(mem.data(), sizeof(mem.data()[0]), mem.size());
}

template <typename T>
inline void AsanUnpoison(const T& mem) {
  rtc_AsanUnpoison(mem.data(), sizeof(mem.data()[0]), mem.size());
}

template <typename T>
inline void MsanMarkUninitialized(const T& mem) {
  rtc_MsanMarkUninitialized(mem.data(), sizeof(mem.data()[0]), mem.size());
}

template <typename T>
inline T MsanUninitialized(T t) {
#if RTC_HAS_MSAN
  // TODO(bugs.webrtc.org/8762): Switch to std::is_trivially_copyable when it
  // becomes available in downstream projects.
  static_assert(sanitizer_impl::IsTriviallyCopyable<T>(), "");
#endif
  rtc_MsanMarkUninitialized(&t, sizeof(T), 1);
  return t;
}

template <typename T>
inline void MsanCheckInitialized(const T& mem) {
  rtc_MsanCheckInitialized(mem.data(), sizeof(mem.data()[0]), mem.size());
}

}  // namespace rtc

#endif  // __cplusplus

#endif  // RTC_BASE_SANITIZER_H_
