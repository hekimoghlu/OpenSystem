/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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
/**
 * @file
 * Includes and defines some default data types.
 */

#ifndef ZYCORE_TYPES_H
#define ZYCORE_TYPES_H

#include "ZycoreDefines.h"
#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

/* ============================================================================================== */
/* Integer types                                                                                  */
/* ============================================================================================== */

#if defined(ZYAN_NO_LIBC) || \
    (defined(ZYAN_MSVC) && defined(ZYAN_KERNEL)) // The WDK LibC lacks stdint.h.
    // No LibC mode, use compiler built-in types / macros.
#   if defined(ZYAN_MSVC) || defined(ZYAN_ICC)
        typedef unsigned __int8  ZyanU8;
        typedef unsigned __int16 ZyanU16;
        typedef unsigned __int32 ZyanU32;
        typedef unsigned __int64 ZyanU64;
        typedef   signed __int8  ZyanI8;
        typedef   signed __int16 ZyanI16;
        typedef   signed __int32 ZyanI32;
        typedef   signed __int64 ZyanI64;
#       if _WIN64
           typedef ZyanU64       ZyanUSize;
           typedef ZyanI64       ZyanISize;
           typedef ZyanU64       ZyanUPointer;
           typedef ZyanI64       ZyanIPointer;
#       else
           typedef ZyanU32       ZyanUSize;
           typedef ZyanI32       ZyanISize;
           typedef ZyanU32       ZyanUPointer;
           typedef ZyanI32       ZyanIPointer;
#       endif
#   elif defined(ZYAN_GNUC)
        typedef __UINT8_TYPE__   ZyanU8;
        typedef __UINT16_TYPE__  ZyanU16;
        typedef __UINT32_TYPE__  ZyanU32;
        typedef __UINT64_TYPE__  ZyanU64;
        typedef __INT8_TYPE__    ZyanI8;
        typedef __INT16_TYPE__   ZyanI16;
        typedef __INT32_TYPE__   ZyanI32;
        typedef __INT64_TYPE__   ZyanI64;
        typedef __SIZE_TYPE__    ZyanUSize;
        typedef __PTRDIFF_TYPE__ ZyanISize;
        typedef __UINTPTR_TYPE__ ZyanUPointer;
        typedef __INTPTR_TYPE__  ZyanIPointer;
#   else
#       error "Unsupported compiler for no-libc mode."
#   endif
#else
    // If is LibC present, we use stdint types.
#   include <stdint.h>
#   include <stddef.h>
    typedef uint8_t   ZyanU8;
    typedef uint16_t  ZyanU16;
    typedef uint32_t  ZyanU32;
    typedef uint64_t  ZyanU64;
    typedef int8_t    ZyanI8;
    typedef int16_t   ZyanI16;
    typedef int32_t   ZyanI32;
    typedef int64_t   ZyanI64;
    typedef size_t    ZyanUSize;
    typedef ptrdiff_t ZyanISize;
    typedef uintptr_t ZyanUPointer;
    typedef intptr_t  ZyanIPointer;
#endif

// Verify size assumptions.
ZYAN_STATIC_ASSERT(sizeof(ZyanU8      ) == 1            );
ZYAN_STATIC_ASSERT(sizeof(ZyanU16     ) == 2            );
ZYAN_STATIC_ASSERT(sizeof(ZyanU32     ) == 4            );
ZYAN_STATIC_ASSERT(sizeof(ZyanU64     ) == 8            );
ZYAN_STATIC_ASSERT(sizeof(ZyanI8      ) == 1            );
ZYAN_STATIC_ASSERT(sizeof(ZyanI16     ) == 2            );
ZYAN_STATIC_ASSERT(sizeof(ZyanI32     ) == 4            );
ZYAN_STATIC_ASSERT(sizeof(ZyanI64     ) == 8            );
ZYAN_STATIC_ASSERT(sizeof(ZyanUSize   ) == sizeof(void*)); // TODO: This one is incorrect!
ZYAN_STATIC_ASSERT(sizeof(ZyanISize   ) == sizeof(void*)); // TODO: This one is incorrect!
ZYAN_STATIC_ASSERT(sizeof(ZyanUPointer) == sizeof(void*));
ZYAN_STATIC_ASSERT(sizeof(ZyanIPointer) == sizeof(void*));

// Verify signedness assumptions (relies on size checks above).
ZYAN_STATIC_ASSERT((ZyanI8 )-1 >> 1 < (ZyanI8 )((ZyanU8 )-1 >> 1));
ZYAN_STATIC_ASSERT((ZyanI16)-1 >> 1 < (ZyanI16)((ZyanU16)-1 >> 1));
ZYAN_STATIC_ASSERT((ZyanI32)-1 >> 1 < (ZyanI32)((ZyanU32)-1 >> 1));
ZYAN_STATIC_ASSERT((ZyanI64)-1 >> 1 < (ZyanI64)((ZyanU64)-1 >> 1));

/* ============================================================================================== */
/* Pointer                                                                                        */
/* ============================================================================================== */

/**
 * Defines the `ZyanVoidPointer` data-type.
 */
typedef char* ZyanVoidPointer;

/**
 * Defines the `ZyanConstVoidPointer` data-type.
 */
typedef const void* ZyanConstVoidPointer;

#define ZYAN_NULL ((void*)0)

/* ============================================================================================== */
/* Logic types                                                                                    */
/* ============================================================================================== */

/* ---------------------------------------------------------------------------------------------- */
/* Boolean                                                                                        */
/* ---------------------------------------------------------------------------------------------- */

#define ZYAN_FALSE 0
#define ZYAN_TRUE  1

/**
 * Defines the `ZyanBool` data-type.
 *
 * Represents a default boolean data-type where `0` is interpreted as `false` and all other values
 * as `true`.
 */
typedef ZyanU8 ZyanBool;

/* ---------------------------------------------------------------------------------------------- */
/* Ternary                                                                                        */
/* ---------------------------------------------------------------------------------------------- */

/**
 * Defines the `ZyanTernary` data-type.
 *
 * The `ZyanTernary` is a balanced ternary type that uses three truth values indicating `true`,
 * `false` and an indeterminate third value.
 */
typedef ZyanI8 ZyanTernary;

#define ZYAN_TERNARY_FALSE    (-1)
#define ZYAN_TERNARY_UNKNOWN  0x00
#define ZYAN_TERNARY_TRUE     0x01

/* ============================================================================================== */
/* String types                                                                                   */
/* ============================================================================================== */

/* ---------------------------------------------------------------------------------------------- */
/* C-style strings                                                                                */
/* ---------------------------------------------------------------------------------------------- */

/**
 * Defines the `ZyanCharPointer` data-type.
 *
 * This type is most often used to represent null-terminated strings aka. C-style strings.
 */
typedef char* ZyanCharPointer;

/**
 * Defines the `ZyanConstCharPointer` data-type.
 *
 * This type is most often used to represent null-terminated strings aka. C-style strings.
 */
typedef const char* ZyanConstCharPointer;

/* ---------------------------------------------------------------------------------------------- */

/* ============================================================================================== */

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif /* ZYCORE_TYPES_H */
