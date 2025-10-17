/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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
#ifndef _I386__PARAM_H_
#define _I386__PARAM_H_

#if defined (__i386__) || defined (__x86_64__)

#include <i386/_types.h>

/*
 * Round p (pointer or byte index) up to a correctly-aligned value for all
 * data types (int, long, ...).   The result is unsigned int and must be
 * cast to any desired pointer type.
 */
#define __DARWIN_ALIGNBYTES     (sizeof(__darwin_size_t) - 1)
#define __DARWIN_ALIGN(p)       ((__darwin_size_t)((__darwin_size_t)(p) + __DARWIN_ALIGNBYTES) &~ __DARWIN_ALIGNBYTES)

#define      __DARWIN_ALIGNBYTES32     (sizeof(__uint32_t) - 1)
#define       __DARWIN_ALIGN32(p)       ((__darwin_size_t)((__darwin_size_t)(p) + __DARWIN_ALIGNBYTES32) &~ __DARWIN_ALIGNBYTES32)

#endif /* defined (__i386__) || defined (__x86_64__) */

#endif /* _I386__PARAM_H_ */
