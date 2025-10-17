/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
#ifndef _STDINT_H
#define _STDINT_H

#include <sys/cdefs.h>

#include <bits/wchar_limits.h>
#include <stddef.h>

typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef short __int16_t;
typedef unsigned short __uint16_t;
typedef int __int32_t;
typedef unsigned int __uint32_t;
#if defined(__LP64__)
typedef long __int64_t;
typedef unsigned long __uint64_t;
#else
typedef long long __int64_t;
typedef unsigned long long __uint64_t;
#endif

#if defined(__LP64__)
typedef long __intptr_t;
typedef unsigned long __uintptr_t;
#else
typedef int __intptr_t;
typedef unsigned int __uintptr_t;
#endif

typedef __int8_t      int8_t;
typedef __uint8_t     uint8_t;

typedef __int16_t     int16_t;
typedef __uint16_t    uint16_t;

typedef __int32_t     int32_t;
typedef __uint32_t    uint32_t;

typedef __int64_t     int64_t;
typedef __uint64_t    uint64_t;

typedef __intptr_t    intptr_t;
typedef __uintptr_t   uintptr_t;

typedef int8_t        int_least8_t;
typedef uint8_t       uint_least8_t;

typedef int16_t       int_least16_t;
typedef uint16_t      uint_least16_t;

typedef int32_t       int_least32_t;
typedef uint32_t      uint_least32_t;

typedef int64_t       int_least64_t;
typedef uint64_t      uint_least64_t;

typedef int8_t        int_fast8_t;
typedef uint8_t       uint_fast8_t;

typedef int64_t       int_fast64_t;
typedef uint64_t      uint_fast64_t;

#if defined(__LP64__)
typedef int64_t       int_fast16_t;
typedef uint64_t      uint_fast16_t;
typedef int64_t       int_fast32_t;
typedef uint64_t      uint_fast32_t;
#else
typedef int32_t       int_fast16_t;
typedef uint32_t      uint_fast16_t;
typedef int32_t       int_fast32_t;
typedef uint32_t      uint_fast32_t;
#endif

typedef uint64_t      uintmax_t;
typedef int64_t       intmax_t;

/* Keep the kernel from trying to define these types... */
#define __BIT_TYPES_DEFINED__

#define INT8_C(c)         c
#define INT_LEAST8_C(c)   INT8_C(c)
#define INT_FAST8_C(c)    INT8_C(c)

#define UINT8_C(c)        c
#define UINT_LEAST8_C(c)  UINT8_C(c)
#define UINT_FAST8_C(c)   UINT8_C(c)

#define INT16_C(c)        c
#define INT_LEAST16_C(c)  INT16_C(c)
#define INT_FAST16_C(c)   INT32_C(c)

#define UINT16_C(c)       c
#define UINT_LEAST16_C(c) UINT16_C(c)
#define UINT_FAST16_C(c)  UINT32_C(c)
#define INT32_C(c)        c
#define INT_LEAST32_C(c)  INT32_C(c)
#define INT_FAST32_C(c)   INT32_C(c)

#define UINT32_C(c)       c ## U
#define UINT_LEAST32_C(c) UINT32_C(c)
#define UINT_FAST32_C(c)  UINT32_C(c)
#define INT_LEAST64_C(c)  INT64_C(c)
#define INT_FAST64_C(c)   INT64_C(c)

#define UINT_LEAST64_C(c) UINT64_C(c)
#define UINT_FAST64_C(c)  UINT64_C(c)

#define INTMAX_C(c)       INT64_C(c)
#define UINTMAX_C(c)      UINT64_C(c)

#if defined(__LP64__)
#  define INT64_C(c)      c ## L
#  define UINT64_C(c)     c ## UL
#  define INTPTR_C(c)     INT64_C(c)
#  define UINTPTR_C(c)    UINT64_C(c)
#  define PTRDIFF_C(c)    INT64_C(c)
#else
#  define INT64_C(c)      c ## LL
#  define UINT64_C(c)     c ## ULL
#  define INTPTR_C(c)     INT32_C(c)
#  define UINTPTR_C(c)    UINT32_C(c)
#  define PTRDIFF_C(c)    INT32_C(c)
#endif

#define INT8_MIN         (-128)
#define INT8_MAX         (127)
#define INT_LEAST8_MIN   INT8_MIN
#define INT_LEAST8_MAX   INT8_MAX
#define INT_FAST8_MIN    INT8_MIN
#define INT_FAST8_MAX    INT8_MAX

#define UINT8_MAX        (255)
#define UINT_LEAST8_MAX  UINT8_MAX
#define UINT_FAST8_MAX   UINT8_MAX

#define INT16_MIN        (-32768)
#define INT16_MAX        (32767)
#define INT_LEAST16_MIN  INT16_MIN
#define INT_LEAST16_MAX  INT16_MAX
#define INT_FAST16_MIN   INT32_MIN
#define INT_FAST16_MAX   INT32_MAX

#define UINT16_MAX       (65535)
#define UINT_LEAST16_MAX UINT16_MAX
#define UINT_FAST16_MAX  UINT32_MAX

#define INT32_MIN        (-2147483647-1)
#define INT32_MAX        (2147483647)
#define INT_LEAST32_MIN  INT32_MIN
#define INT_LEAST32_MAX  INT32_MAX
#define INT_FAST32_MIN   INT32_MIN
#define INT_FAST32_MAX   INT32_MAX

#define UINT32_MAX       (4294967295U)
#define UINT_LEAST32_MAX UINT32_MAX
#define UINT_FAST32_MAX  UINT32_MAX

#define INT64_MIN        (INT64_C(-9223372036854775807)-1)
#define INT64_MAX        (INT64_C(9223372036854775807))
#define INT_LEAST64_MIN  INT64_MIN
#define INT_LEAST64_MAX  INT64_MAX
#define INT_FAST64_MIN   INT64_MIN
#define INT_FAST64_MAX   INT64_MAX
#define UINT64_MAX       (UINT64_C(18446744073709551615))

#define UINT_LEAST64_MAX UINT64_MAX
#define UINT_FAST64_MAX  UINT64_MAX

#define INTMAX_MIN       INT64_MIN
#define INTMAX_MAX       INT64_MAX
#define UINTMAX_MAX      UINT64_MAX

#define SIG_ATOMIC_MAX   INT32_MAX
#define SIG_ATOMIC_MIN   INT32_MIN

#if defined(__WINT_UNSIGNED__)
#  define WINT_MAX       UINT32_MAX
#  define WINT_MIN       0
#else
#  define WINT_MAX       INT32_MAX
#  define WINT_MIN       INT32_MIN
#endif

#if defined(__LP64__)
#  define INTPTR_MIN     INT64_MIN
#  define INTPTR_MAX     INT64_MAX
#  define UINTPTR_MAX    UINT64_MAX
#  define PTRDIFF_MIN    INT64_MIN
#  define PTRDIFF_MAX    INT64_MAX
#  define SIZE_MAX       UINT64_MAX
#else
#  define INTPTR_MIN     INT32_MIN
#  define INTPTR_MAX     INT32_MAX
#  define UINTPTR_MAX    UINT32_MAX
#  define PTRDIFF_MIN    INT32_MIN
#  define PTRDIFF_MAX    INT32_MAX
#  define SIZE_MAX       UINT32_MAX
#endif

#endif /* _STDINT_H */
