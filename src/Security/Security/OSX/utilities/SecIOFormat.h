/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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
#ifndef _SECIOFORMAT_H_
#define _SECIOFORMAT_H_
// TODO: Name this file SecType.h?  To match inttype.h?

#include <inttypes.h>

// MARK: Guidlines and Examples

/*  Tips for using printf and CFStringCreateWithFormat style functions.

    Avoid using casts in arguments to these functions like the plague.  If you
    have to use them, think again.  You are probably wrong and you are writing
    non portable code.  Instead try following this pattern:

    Type        Format String       Variants
    size_t      "%zu"
    ssize_t     "%zd"
    ptrdiff_t   "%td"               printf("array_len: %td", pos - begin)
    OSStatus    "%"PRIdOSStatus     printf("%"PRIxCFIndex" returned", status)
    CFIndex     "%"PRIdCFIndex      printf("ar[%"PRIdCFIndex"]=%p", ix, CFArrayGetValueAtIndex(ar, ix))
    sint64_t    "%"PRId64
    uint64_t    "%"PRIu64           printf("sqlite3_rowid: %"PRIu64, rowid)
    sint32_t    "%"PRId32
    uint32_t    "%"PRIu32
    uint8_t     "%"PRIx8

    All of the above examples also work inside a CFSTR(). For example:
    CFStringAppendFormat(ms, NULL, CFSTR("ar[%"PRIdCFIndex"]=%p"), ix, CFArrayGetValueAtIndex(ar, ix))
    Also you can use any of d i o u x X where appropriate for some of these types
    although u x X are for unsigned types only.

    Try to avoid using these types unless you know what you are doing because
    they can lead to portability issues on different flavors of 32 and 64 bit
    platforms:

    int         "%d"
    long        "%ld"
    long long   "%lld"
 */

// MARK: CFIndex printing support

// Note that CFIndex is signed so the u variants won't work.
#ifdef __LLP64__
#  define PRIdCFIndex    "lld"
#  define PRIiCFIndex    "lli"
#  define PRIoCFIndex    "llo"
#  define PRIuCFIndex    "llu"
#  define PRIxCFIndex    "llx"
#  define PRIXCFIndex    "llX"
#else
#  define PRIdCFIndex    "ld"
#  define PRIiCFIndex    "li"
#  define PRIoCFIndex    "lo"
#  define PRIuCFIndex    "lu"
#  define PRIxCFIndex    "lx"
#  define PRIXCFIndex    "lX"
#endif

// MARK: OSStatus printing support

// Note that OSStatus is signed so the u variants won't work.
#ifdef __LP64__
#  define PRIdOSStatus    "d"
#  define PRIiOSStatus    "i"
#  define PRIoOSStatus    "o"
#  define PRIuOSStatus    "u"
#  define PRIxOSStatus    "x"
#  define PRIXOSStatus    "X"
#else
#  define PRIdOSStatus    "ld"
#  define PRIiOSStatus    "li"
#  define PRIoOSStatus    "lo"
#  define PRIuOSStatus    "lu"
#  define PRIxOSStatus    "lx"
#  define PRIXOSStatus    "lX"
#endif

#endif
