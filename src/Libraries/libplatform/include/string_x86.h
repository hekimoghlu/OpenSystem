/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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
#ifndef _STRING_X86_H
#define _STRING_X86_H

#include <Availability.h>

#if defined(__x86_64__)

__BEGIN_DECLS
/* These SSE variants have the same behavior as their original functions.
 * SSE instructions are used in these variants instead of best possible
 * implementation.
 */
__OSX_AVAILABLE(10.16) __IOS_UNAVAILABLE __TVOS_UNAVAILABLE __WATCHOS_UNAVAILABLE
void	*memmove_sse_np(void *__dst, const void *__src, size_t __len);

__OSX_AVAILABLE(10.16) __IOS_UNAVAILABLE __TVOS_UNAVAILABLE __WATCHOS_UNAVAILABLE
void	*memset_sse_np(void *__b, int __c, size_t __len);

__OSX_AVAILABLE(10.16) __IOS_UNAVAILABLE __TVOS_UNAVAILABLE __WATCHOS_UNAVAILABLE
void	 bzero_sse_np(void *, size_t);

__OSX_AVAILABLE(10.16) __IOS_UNAVAILABLE __TVOS_UNAVAILABLE __WATCHOS_UNAVAILABLE
void     memset_pattern4_sse_np(void *__b, const void *__pattern4, size_t __len);

__OSX_AVAILABLE(10.16) __IOS_UNAVAILABLE __TVOS_UNAVAILABLE __WATCHOS_UNAVAILABLE
void     memset_pattern8_sse_np(void *__b, const void *__pattern8, size_t __len);

__OSX_AVAILABLE(10.16) __IOS_UNAVAILABLE __TVOS_UNAVAILABLE __WATCHOS_UNAVAILABLE
void     memset_pattern16_sse_np(void *__b, const void *__pattern16, size_t __len);
__END_DECLS

#endif /* __x86_64__ */

#endif /* _STRING_X86_H */
