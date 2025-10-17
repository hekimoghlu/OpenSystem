/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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
#ifndef WKDeclarationSpecifiers_h
#define WKDeclarationSpecifiers_h

#ifndef __has_declspec_attribute
#define __has_declspec_attribute(x) 0
#endif

#undef WK_EXPORT
#if defined(WK_NO_EXPORT)
#define WK_EXPORT
#elif defined(WIN32) || defined(_WIN32) || defined(__CC_ARM) || defined(__ARMCC__) || (__has_declspec_attribute(dllimport) && __has_declspec_attribute(dllexport))
#if defined(BUILDING_WebKit)
#define WK_EXPORT __declspec(dllexport)
#else
#define WK_EXPORT __declspec(dllimport)
#endif /* defined(BUILDING_WebKit) */
#elif defined(__GNUC__)
#define WK_EXPORT __attribute__((visibility("default")))
#else /* !defined(WK_NO_EXPORT) */
#define WK_EXPORT
#endif /* defined(WK_NO_EXPORT) */

#if !defined(WK_INLINE)
#if defined(__cplusplus)
#define WK_INLINE static inline
#elif defined(__GNUC__)
#define WK_INLINE static __inline__
#else
#define WK_INLINE static
#endif
#endif /* !defined(WK_INLINE) */

#ifndef __has_extension
#define __has_extension(x) 0
#endif

#if __has_extension(enumerator_attributes) && __has_extension(attribute_unavailable_with_message)
#define WK_C_DEPRECATED(message) __attribute__((deprecated(message)))
#else
#define WK_C_DEPRECATED(message)
#endif

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#if __has_attribute(unavailable)
#define WK_UNAVAILABLE __attribute__((unavailable))
#else
#define WK_UNAVAILABLE
#endif

#endif /* WKDeclarationSpecifiers_h */
