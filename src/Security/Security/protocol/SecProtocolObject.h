/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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
#ifndef SecProtocolObject_h
#define SecProtocolObject_h

#include <sys/cdefs.h>
#include <os/object.h>

#if OS_OBJECT_USE_OBJC
#  define SEC_OBJECT_DECL(type) OS_OBJECT_DECL(type)
#else // OS_OBJECT_USE_OBJC
#  define SEC_OBJECT_DECL(type)                \
struct type;                        \
typedef    struct type *type##_t
#endif // OS_OBJECT_USE_OBJC

#if __has_feature(assume_nonnull)
#  define SEC_ASSUME_NONNULL_BEGIN _Pragma("clang assume_nonnull begin")
#  define SEC_ASSUME_NONNULL_END   _Pragma("clang assume_nonnull end")
#else // !__has_feature(assume_nonnull)
#  define SEC_ASSUME_NONNULL_BEGIN
#  define SEC_ASSUME_NONNULL_END
#endif // !__has_feature(assume_nonnull)

#if defined(__OBJC__) && __has_attribute(ns_returns_retained)
#  define SEC_RETURNS_RETAINED __attribute__((__ns_returns_retained__))
#else // __OBJC__ && ns_returns_retained
#  define SEC_RETURNS_RETAINED
#endif // __OBJC__ && ns_returns_retained

#if !OS_OBJECT_USE_OBJC_RETAIN_RELEASE
__BEGIN_DECLS
__attribute__((visibility("default"))) void *sec_retain(void *obj);
__attribute__((visibility("default"))) void sec_release(void *obj);
__END_DECLS
#else // !OS_OBJECT_USE_OBJC_RETAIN_RELEASE
#undef sec_retain
#undef sec_release
#define sec_retain(object) [(object) retain]
#define sec_release(object) [(object) release]
#endif // !OS_OBJECT_USE_OBJC_RETAIN_RELEASE

#ifndef SEC_OBJECT_IMPL
/*!
 * A `sec_object` is a generic, ARC-able type wrapper for common CoreFoundation Security types.
 */
SEC_OBJECT_DECL(sec_object);
#endif // !SEC_OBJECT_IMPL

#endif // SecProtocolObject_h
