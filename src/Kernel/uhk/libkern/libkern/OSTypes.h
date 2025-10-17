/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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
#if !defined(KERNEL)
#include <MacTypes.h>
#endif  /* !KERNEL */

#ifndef _OS_OSTYPES_H
#define _OS_OSTYPES_H

#define OSTYPES_K64_REV         2

typedef unsigned int       UInt;
typedef signed int         SInt;

#if defined(KERNEL)

typedef unsigned char      UInt8;
typedef unsigned short     UInt16;
#if __LP64__
typedef unsigned int       UInt32;
#else
typedef unsigned long      UInt32;
#endif
typedef unsigned long long UInt64;
#if             defined(__BIG_ENDIAN__)
typedef struct __attribute__((deprecated)) UnsignedWide {
	UInt32          hi;
	UInt32          lo;
}                                                       UnsignedWide __attribute__((deprecated));
#elif           defined(__LITTLE_ENDIAN__)
typedef struct __attribute__((deprecated)) UnsignedWide {
	UInt32          lo;
	UInt32          hi;
}                                                       UnsignedWide __attribute__((deprecated));
#else
#error Unknown endianess.
#endif

typedef signed char        SInt8;
typedef signed short       SInt16;
#if __LP64__
typedef signed int         SInt32;
#else
typedef signed long        SInt32;
#endif
typedef signed long long   SInt64;

typedef SInt32                          OSStatus;

#ifndef ABSOLUTETIME_SCALAR_TYPE
#define ABSOLUTETIME_SCALAR_TYPE    1
#endif
typedef UInt64          AbsoluteTime;

typedef UInt32                          OptionBits __attribute__((deprecated));

#if defined(__LP64__)
/*
 * Use intrinsic boolean types for the LP64 kernel, otherwise maintain
 * source and binary backward compatibility.  This attempts to resolve
 * the "(x == true)" vs. "(x)" conditional issue.
 */
#ifdef __cplusplus
typedef bool Boolean;
#else   /* !__cplusplus */
#if defined(__STDC_VERSION__) && ((__STDC_VERSION__ - 199901L) > 0L)
/* only use this if we are sure we are using a c99 compiler */
typedef _Bool Boolean;
#else   /* !c99 */
/* Fall back to previous definition unless c99 */
typedef unsigned char Boolean;
#endif  /* !c99 */
#endif  /* !__cplusplus */
#else   /* !__LP64__ */
typedef unsigned char Boolean;
#endif  /* !__LP64__ */

#endif  /* KERNEL */

#include <sys/_types/_os_inline.h>

#endif /* _OS_OSTYPES_H */
