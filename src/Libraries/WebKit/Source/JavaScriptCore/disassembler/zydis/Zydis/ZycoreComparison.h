/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
 * Defines prototypes of general-purpose comparison functions.
 */

#ifndef ZYCORE_COMPARISON_H
#define ZYCORE_COMPARISON_H

#include "ZycoreDefines.h"
#include "ZycoreTypes.h"
#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================== */
/* Enums and types                                                                                */
/* ============================================================================================== */

/**
 * Defines the `ZyanEqualityComparison` function prototype.
 *
 * @param   left    A pointer to the first element.
 * @param   right   A pointer to the second element.
 *
 * @return  This function should return `ZYAN_TRUE` if the `left` element equals the `right` one
 *          or `ZYAN_FALSE`, if not.
 */
typedef ZyanBool (*ZyanEqualityComparison)(const void* left, const void* right);

/**
 * Defines the `ZyanComparison` function prototype.
 *
 * @param   left    A pointer to the first element.
 * @param   right   A pointer to the second element.
 *
 * @return  This function should return values in the following range:
 *          `left == right -> result == 0`
 *          `left <  right -> result  < 0`
 *          `left >  right -> result  > 0`
 */
typedef ZyanI32 (*ZyanComparison)(const void* left, const void* right);

/* ============================================================================================== */
/* Macros                                                                                         */
/* ============================================================================================== */

/* ---------------------------------------------------------------------------------------------- */
/* Equality comparison functions                                                                  */
/* ---------------------------------------------------------------------------------------------- */

/**
 * Declares a generic equality comparison function for an integral data-type.
 *
 * @param   name    The name of the function.
 * @param   type    The name of the integral data-type.
 */
#define ZYAN_DECLARE_EQUALITY_COMPARISON(name, type) \
    ZyanBool name(const type* left, const type* right) \
    { \
        ZYAN_ASSERT(left); \
        ZYAN_ASSERT(right); \
        \
        return (*left == *right) ? ZYAN_TRUE : ZYAN_FALSE; \
    }

/**
 * Declares a generic equality comparison function that compares a single integral
 *          data-type field of a struct.
 *
 * @param   name        The name of the function.
 * @param   type        The name of the integral data-type.
 * @param   field_name  The name of the struct field.
 */
#define ZYAN_DECLARE_EQUALITY_COMPARISON_FOR_FIELD(name, type, field_name) \
    ZyanBool name(const type* left, const type* right) \
    { \
        ZYAN_ASSERT(left); \
        ZYAN_ASSERT(right); \
        \
        return (left->field_name == right->field_name) ? ZYAN_TRUE : ZYAN_FALSE; \
    }

/* ---------------------------------------------------------------------------------------------- */
/* Comparison functions                                                                           */
/* ---------------------------------------------------------------------------------------------- */

/**
 * Declares a generic comparison function for an integral data-type.
 *
 * @param   name    The name of the function.
 * @param   type    The name of the integral data-type.
 */
#define ZYAN_DECLARE_COMPARISON(name, type) \
    ZyanI32 name(const type* left, const type* right) \
    { \
        ZYAN_ASSERT(left); \
        ZYAN_ASSERT(right); \
        \
        if (*left < *right) \
        { \
            return -1; \
        } \
        if (*left > *right) \
        { \
            return  1; \
        } \
        return 0; \
    }

/**
 * Declares a generic comparison function that compares a single integral data-type field
 *          of a struct.
 *
 * @param   name        The name of the function.
 * @param   type        The name of the integral data-type.
 * @param   field_name  The name of the struct field.
 */
#define ZYAN_DECLARE_COMPARISON_FOR_FIELD(name, type, field_name) \
    ZyanI32 name(const type* left, const type* right) \
    { \
        ZYAN_ASSERT(left); \
        ZYAN_ASSERT(right); \
        \
        if (left->field_name < right->field_name) \
        { \
            return -1; \
        } \
        if (left->field_name > right->field_name) \
        { \
            return  1; \
        } \
        return 0; \
    }

 /* ---------------------------------------------------------------------------------------------- */

/* ============================================================================================== */
/* Exported functions                                                                             */
/* ============================================================================================== */

/* ---------------------------------------------------------------------------------------------- */
/* Default equality comparison functions                                                          */
/* ---------------------------------------------------------------------------------------------- */

/**
 * Defines a default equality comparison function for pointer values.
 *
 * @param   left    A pointer to the first value.
 * @param   right   A pointer to the second value.
 *
 * @return  Returns `ZYAN_TRUE` if the `left` value equals the `right` one or `ZYAN_FALSE`, if
 *          not.
 */
ZYAN_INLINE ZYAN_DECLARE_EQUALITY_COMPARISON(ZyanEqualsPointer, void* const)

/**
 * Defines a default equality comparison function for `ZyanBool` values.
 *
 * @param   left    A pointer to the first value.
 * @param   right   A pointer to the second value.
 *
 * @return  Returns `ZYAN_TRUE` if the `left` value equals the `right` one or `ZYAN_FALSE`, if
 *          not.
 */
ZYAN_INLINE ZYAN_DECLARE_EQUALITY_COMPARISON(ZyanEqualsBool, ZyanBool)

/**
 * Defines a default equality comparison function for 8-bit numeric values.
 *
 * @param   left    A pointer to the first value.
 * @param   right   A pointer to the second value.
 *
 * @return  Returns `ZYAN_TRUE` if the `left` value equals the `right` one or `ZYAN_FALSE`, if
 *          not.
 */
ZYAN_INLINE ZYAN_DECLARE_EQUALITY_COMPARISON(ZyanEqualsNumeric8, ZyanU8)

/**
 * Defines a default equality comparison function for 16-bit numeric values.
 *
 * @param   left    A pointer to the first value.
 * @param   right   A pointer to the second value.
 *
 * @return  Returns `ZYAN_TRUE` if the `left` value equals the `right` one or `ZYAN_FALSE`, if
 *          not.
 */
ZYAN_INLINE ZYAN_DECLARE_EQUALITY_COMPARISON(ZyanEqualsNumeric16, ZyanU16)

/**
 * Defines a default equality comparison function for 32-bit numeric values.
 *
 * @param   left    A pointer to the first value.
 * @param   right   A pointer to the second value.
 *
 * @return  Returns `ZYAN_TRUE` if the `left` value equals the `right` one or `ZYAN_FALSE`, if
 *          not.
 */
ZYAN_INLINE ZYAN_DECLARE_EQUALITY_COMPARISON(ZyanEqualsNumeric32, ZyanU32)

/**
 * Defines a default equality comparison function for 64-bit numeric values.
 *
 * @param   left    A pointer to the first value.
 * @param   right   A pointer to the second value.
 *
 * @return  Returns `ZYAN_TRUE` if the `left` value equals the `right` one or `ZYAN_FALSE`, if
 *          not.
 */
ZYAN_INLINE ZYAN_DECLARE_EQUALITY_COMPARISON(ZyanEqualsNumeric64, ZyanU64)

/* ---------------------------------------------------------------------------------------------- */
/* Default comparison functions                                                                   */
/* ---------------------------------------------------------------------------------------------- */

/**
 * Defines a default comparison function for pointer values.
 *
 * @param   left    A pointer to the first value.
 * @param   right   A pointer to the second value.
 *
 * @return  Returns `0` if the `left` value equals the `right` one, `-1` if the `left` value is
 *          less than the `right` one, or `1` if the `left` value is greater than the `right` one.
 */
ZYAN_INLINE ZYAN_DECLARE_COMPARISON(ZyanComparePointer, void* const)

/**
 * Defines a default comparison function for `ZyanBool` values.
 *
 * @param   left    A pointer to the first value.
 * @param   right   A pointer to the second value.
 *
 * @return  Returns `0` if the `left` value equals the `right` one, `-1` if the `left` value is
 *          less than the `right` one, or `1` if the `left` value is greater than the `right` one.
 */
ZYAN_INLINE ZYAN_DECLARE_COMPARISON(ZyanCompareBool, ZyanBool)

/**
 * Defines a default comparison function for 8-bit numeric values.
 *
 * @param   left    A pointer to the first value.
 * @param   right   A pointer to the second value.
 *
 * @return  Returns `0` if the `left` value equals the `right` one, `-1` if the `left` value is
 *          less than the `right` one, or `1` if the `left` value is greater than the `right` one.
 */
ZYAN_INLINE ZYAN_DECLARE_COMPARISON(ZyanCompareNumeric8, ZyanU8)

/**
 * Defines a default comparison function for 16-bit numeric values.
 *
 * @param   left    A pointer to the first value.
 * @param   right   A pointer to the second value.
 *
 * @return  Returns `0` if the `left` value equals the `right` one, `-1` if the `left` value is
 *          less than the `right` one, or `1` if the `left` value is greater than the `right` one.
 */
ZYAN_INLINE ZYAN_DECLARE_COMPARISON(ZyanCompareNumeric16, ZyanU16)

/**
 * Defines a default comparison function for 32-bit numeric values.
 *
 * @param   left    A pointer to the first value.
 * @param   right   A pointer to the second value.
 *
 * @return  Returns `0` if the `left` value equals the `right` one, `-1` if the `left` value is
 *          less than the `right` one, or `1` if the `left` value is greater than the `right` one.
 */
ZYAN_INLINE ZYAN_DECLARE_COMPARISON(ZyanCompareNumeric32, ZyanU32)

/**
 * Defines a default comparison function for 64-bit numeric values.
 *
 * @param   left    A pointer to the first value.
 * @param   right   A pointer to the second value.
 *
 * @return  Returns `0` if the `left` value equals the `right` one, `-1` if the `left` value is
 *          less than the `right` one, or `1` if the `left` value is greater than the `right` one.
 */
ZYAN_INLINE ZYAN_DECLARE_COMPARISON(ZyanCompareNumeric64, ZyanU64)

/* ---------------------------------------------------------------------------------------------- */

/* ============================================================================================== */

#ifdef __cplusplus
}
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif /* ZYCORE_COMPARISON_H */
