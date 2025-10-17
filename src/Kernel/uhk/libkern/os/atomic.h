/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#ifndef __OS_ATOMIC_H__
#define __OS_ATOMIC_H__

/*!
 * @file <os/atomic.h>
 *
 * @brief
 * Small header that helps write code that works with both C11 and C++11,
 * or pre-C11 type declarations.
 *
 * @discussion
 * The macros below allow to write code like this, that can be put in a header
 * and will work with both C11 and C++11:
 *
 * <code>
 * struct old_type {
 *     int atomic_field;
 * } old_variable;
 *
 * os_atomic_std(atomic_fetch_add_explicit)(
 *     os_cast_to_atomic_pointer(&old_variable), 1,
 *     os_atomic_std(memory_order_relaxed));
 * </code>
 */

#include <os/base.h>

#ifndef OS_ATOMIC_USES_CXX
#ifdef KERNEL
#define OS_ATOMIC_USES_CXX 0
#elif defined(__cplusplus) && __cplusplus >= 201103L
#define OS_ATOMIC_USES_CXX 1
#else
#define OS_ATOMIC_USES_CXX 0
#endif
#endif

#if OS_ATOMIC_USES_CXX
#include <atomic>
#define OS_ATOMIC_STD                    std::
#define os_atomic_std(op)                std::op
#define os_atomic(type)                  std::atomic<type> volatile
#define os_cast_to_atomic_pointer(p)     os::cast_to_atomic_pointer(p)
#define os_atomic_basetypeof(p)          decltype(os_cast_to_atomic_pointer(p)->load())
#define os_cast_to_nonatomic_pointer(p)  os::cast_to_nonatomic_pointer(p)
#else /* !OS_ATOMIC_USES_CXX */
#include <stdatomic.h>
#define OS_ATOMIC_STD
#define os_atomic_std(op)                op
#define os_atomic(type)                  type volatile _Atomic
#if __has_ptrcheck
#define os_cast_to_atomic_pointer(p)     (__typeof__(*(p)) volatile _Atomic * __single)(p)
#define os_cast_to_nonatomic_pointer(p)                             \
	_Pragma("clang diagnostic push")                        \
	_Pragma("clang diagnostic ignored \"-Wcast-qual\"")     \
	(os_atomic_basetypeof(p) * __single)(p)                 \
	_Pragma("clang diagnostic pop")
#else /* !__has_ptrcheck */
#define os_cast_to_atomic_pointer(p)     (__typeof__(*(p)) volatile _Atomic *)(uintptr_t)(p)
#define os_cast_to_nonatomic_pointer(p)  (os_atomic_basetypeof(p) *)(uintptr_t)(p)
#endif /* !__has_ptrcheck */
#define os_atomic_basetypeof(p)          __typeof__(atomic_load(os_cast_to_atomic_pointer(p)))

#endif /* !OS_ATOMIC_USES_CXX */

/*!
 * @group Internal implementation details
 *
 * @discussion The functions below are not intended to be used directly.
 */

#if OS_ATOMIC_USES_CXX
#include <type_traits>

namespace os {
template <class T> using remove_volatile_t = typename std::remove_volatile<T>::type;

template <class T>
inline volatile std::atomic<remove_volatile_t<T> > *
cast_to_atomic_pointer(T *v)
{
	return reinterpret_cast<volatile std::atomic<remove_volatile_t<T> > *>(v);
}

template <class T>
inline volatile std::atomic<remove_volatile_t<T> > *
cast_to_atomic_pointer(std::atomic<T> *v)
{
	return reinterpret_cast<volatile std::atomic<remove_volatile_t<T> > *>(v);
}

template <class T>
inline volatile std::atomic<remove_volatile_t<T> > *
cast_to_atomic_pointer(volatile std::atomic<T> *v)
{
	return reinterpret_cast<volatile std::atomic<remove_volatile_t<T> > *>(v);
}

template <class T>
inline remove_volatile_t<T> *
cast_to_nonatomic_pointer(T *v)
{
	return const_cast<remove_volatile_t<T> *>(v);
}

template <class T>
inline remove_volatile_t<T> *
cast_to_nonatomic_pointer(std::atomic<T> *v)
{
	return reinterpret_cast<remove_volatile_t<T> *>(v);
}

template <class T>
inline remove_volatile_t<T> *
cast_to_nonatomic_pointer(volatile std::atomic<T> *v)
{
	auto _v = const_cast<std::atomic<T> *>(v);
	return reinterpret_cast<remove_volatile_t<T> *>(_v);
}
};
#endif /* OS_ATOMIC_USES_CXX */

#endif /* __OS_ATOMIC_H__ */
