/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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

// vim:noexpandtab
#ifndef __TYPES_H__
#define __TYPES_H__

#include <stdint.h>
#include <stdbool.h>

typedef signed char     s8;
typedef unsigned char   u8;
typedef uint16_t        u16;
typedef int16_t         s16;
typedef uint32_t        u32;
typedef uint64_t        u64;
typedef int32_t         s32;
typedef int64_t         s64;

#if defined(__arm64__) || defined(__x86_64__)
typedef u64     un;
typedef s64     sn;
#else
typedef u32     un;
typedef s32     sn;
#endif

#ifndef __DRT_H__
typedef u32     uint;
#endif

#define volatile_read(atom)             (*((volatile typeof(*(atom)) *)(atom)))
#define volatile_write(atom, value)     (*((volatile typeof(*(atom)) *)(atom)) = value)

#endif
