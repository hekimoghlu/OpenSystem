/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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
#ifndef __OS_GENERIC_WIN_BASE__
#define __OS_GENERIC_WIN_BASE__

// Unices provide `roundup` via sys/param.h
#define roundup(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
// Unices provide `MAX` via sys/param.h
#define MAX(a,b) (((a)>(b))?(a):(b))
// Unices provide `MIN` via sys/param.h
#define MIN(a,b) (((a)<(b))?(a):(b))
// Unices provide `howmany` via sys/param.h
#define howmany(x, y)  (((x) + ((y) - 1)) / (y))

typedef int mode_t;
typedef void pthread_attr_t;

#if defined(__cplusplus)
#define __BEGIN_DECLS extern "C" {
#define __END_DECLS }
#else
#define __BEGIN_DECLS
#define __END_DECLS
#endif

#ifndef API_AVAILABLE
#define API_AVAILABLE(...)
#endif
#ifndef API_DEPRECATED
#define API_DEPRECATED(...)
#endif
#ifndef API_UNAVAILABLE
#define API_UNAVAILABLE(...)
#endif
#ifndef API_DEPRECATED_WITH_REPLACEMENT
#define API_DEPRECATED_WITH_REPLACEMENT(...)
#endif

#if !defined(__has_attribute)
#define __has_attribute(attibute) 0
#endif

#if !defined(__has_builtin)
#define __has_builtin(builtin) 0
#endif

#if !defined(__has_feature)
#define __has_feature(feature) 0
#endif

#if __has_builtin(__builtin_expect)
#define OS_EXPECT(expression, value) __builtin_expect((expression), (value))
#else
#define OS_EXPECT(expression, value) (expression)
#endif

#if __has_attribute(__unused__)
#define OS_UNUSED __attribute__((__unused__))
#else
#define OS_UNUSED
#endif

#ifndef os_likely
#define os_likely(expression) OS_EXPECT(!!(expression), 1)
#endif
#ifndef os_unlikely
#define os_unlikely(expression) OS_EXPECT(!!(expression), 0)
#endif

#if __has_feature(assume_nonnull)
#define OS_ASSUME_NONNULL_BEGIN _Pragma("clang assume_nonnull begin")
#define OS_ASSUME_NONNULL_END   _Pragma("clang assume_nonnull end")
#else
#define OS_ASSUME_NONNULL_BEGIN
#define OS_ASSUME_NONNULL_END
#endif

#if __has_builtin(__builtin_assume)
#define OS_COMPILER_CAN_ASSUME(expr) __builtin_assume(expr)
#else
#define OS_COMPILER_CAN_ASSUME(expr) ((void)(expr))
#endif

#if __has_feature(attribute_availability_swift)
// equivalent to __SWIFT_UNAVAILABLE from Availability.h
#define OS_SWIFT_UNAVAILABLE(msg)                                              \
  __attribute__((__availability__(swift, unavailable, message = msg)))
#else
#define OS_SWIFT_UNAVAILABLE(msg)
#endif

#define __OS_STRINGIFY(s) #s
#define OS_STRINGIFY(s) __OS_STRINGIFY(s)

#if __has_feature(objc_fixed_enum) || __has_extension(cxx_strong_enums)
#define OS_ENUM(name, type, ...) typedef enum : type { __VA_ARGS__ } name##_t
#else
#define OS_ENUM(name, type, ...)                                               \
  enum { __VA_ARGS__ };                                                        \
  typedef type name##_t
#endif

#ifdef OS_EXPORT
#undef OS_EXPORT
#endif
#define OS_EXPORT __declspec(dllexport)

#ifdef OS_WARN_RESULT_NEEDS_RELEASE
#undef OS_WARN_RESULT_NEEDS_RELEASE
#endif

#ifdef OS_WARN_RESULT
#undef OS_WARN_RESULT
#endif
#define OS_WARN_RESULT

#ifdef OS_NOTHROW
#undef OS_NOTHROW
#endif
#define OS_NOTHROW

#endif
