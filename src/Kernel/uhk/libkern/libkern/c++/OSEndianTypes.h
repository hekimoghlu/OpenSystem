/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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
// xx-review: are these even used anywhere? Grep turns up squat.


/*!
 * @header OSEndianTypes
 *
 * @abstract
 * C++ inline types for byte-swapping.
 *
 * @discussion
 * The OSEndianTypes consist of a number of types that are used
 * very similarly to the traditional MacOS C scalar integers types,
 * for example, <code>UInt32</code> and <code>SInt32</code>.
 * @copyright 2005 Apple Computer, Inc. All rights reserved.
 * @updated 2005-07-25
 */

// Header doc magic trick for simple documentation
#if 0
/*!
 * @typedef BigUInt16
 * @abstract A Big-endian unsigned integer scalar size 16 - UInt16
 */
typedef class BigUInt16 BigUInt16;

/*!
 * @typedef BigSInt16
 * @abstract A Big-endian signed integer scalar size 16 - SInt16
 */
typedef class BigSInt16 BigSInt16;

/*!
 * @typedef BigUInt32
 * @abstract A Big-endian unsigned integer scalar size 32 - UInt32
 */
typedef class BigUInt32 BigUInt32;

/*!
 * @typedef BigSInt32
 * @abstract A Big-endian signed integer scalar size 32 - SInt32
 */
typedef class BigSInt32 BigSInt32;

/*!
 * @typedef BigUInt64
 * @abstract A Big-endian unsigned integer scalar size 64 - UInt64
 */
typedef class BigUInt64 BigUInt64;

/*!
 * @typedef BigSInt64
 * @abstract A Big-endian signed integer scalar size 64 - SInt64
 */
typedef class BigSInt64 BigSInt64;

/*!
 * @typedef LittleUInt16
 * @abstract A Little-endian unsigned integer scalar size 16 - UInt16
 */
typedef class LittleUInt16 LittleUInt16;

/*!
 * @typedef LittleSInt16
 * @abstract A Little-endian signed integer scalar size 16 - SInt16
 */
typedef class LittleSInt16 LittleSInt16;

/*!
 * @typedef LittleUInt32
 * @abstract A Little-endian unsigned integer scalar size 32 - UInt32
 */
typedef class LittleUInt32 LittleUInt32;

/*!
 * @typedef LittleSInt32
 * @abstract A Little-endian signed integer scalar size 32 - SInt32
 */
typedef class LittleSInt32 LittleSInt32;

/*!
 * @typedef LittleUInt64
 * @abstract A Little-endian unsigned integer scalar size 64 - UInt64
 */
typedef class LittleUInt64 LittleUInt64;

/*!
 * @typedef LittleSInt64
 * @abstract A Little-endian signed integer scalar size 64 - SInt64
 */
typedef class LittleSInt64 LittleSInt64;

#endif /* 0 - headerdoc trick */

#ifndef _OS_OSENDIANHELPER_H
#define _OS_OSENDIANHELPER_H

#if __cplusplus

#include <libkern/OSTypes.h>
#include <libkern/OSByteOrder.h>

// Probably should really be using templates, this is one of the few cases
// where they do make sense.  But as the kernel is not allowed to export
// template based C++ APIs we have to use sophisticated macros instead
#define __OSEndianSignIntSizeDEF(argname, argend, argtype, argsize) {  \
public:                                                                \
    typedef argtype ## argsize        Value;                           \
                                                                       \
private:                                                               \
    typedef UInt ## argsize        UValue;                             \
    UValue mValue;                                                     \
                                                                       \
    void writeValue(Value v) {                                         \
    if (__builtin_constant_p(v))                                       \
	mValue = OSSwapHostTo ## argend ## ConstInt ## argsize(v);     \
    else                                                               \
	OSWrite ## argend ## Int ## argsize(&mValue, 0, (UValue) v);   \
    };                                                                 \
                                                                       \
    Value readValue() const {                                          \
    return (Value) OSRead ## argend ## Int ## argsize(&mValue, 0);     \
    };                                                                 \
                                                                       \
public:                                                                \
    argname() { };                                                     \
                                                                       \
    argname (Value v) { writeValue(v); };                              \
    argname  &operator = (Value v) { writeValue(v); return *this; }    \
                                                                       \
    Value get() const { return readValue(); };                         \
    operator Value () const { return readValue(); };                   \
}

class BigUInt16    __OSEndianSignIntSizeDEF(BigUInt16, Big, UInt, 16);
class BigSInt16    __OSEndianSignIntSizeDEF(BigSInt16, Big, SInt, 16);
class BigUInt32    __OSEndianSignIntSizeDEF(BigUInt32, Big, UInt, 32);
class BigSInt32    __OSEndianSignIntSizeDEF(BigSInt32, Big, SInt, 32);
class BigUInt64    __OSEndianSignIntSizeDEF(BigUInt64, Big, UInt, 64);
class BigSInt64    __OSEndianSignIntSizeDEF(BigSInt64, Big, SInt, 64);
class LittleUInt16 __OSEndianSignIntSizeDEF(LittleUInt16, Little, UInt, 16);
class LittleSInt16 __OSEndianSignIntSizeDEF(LittleSInt16, Little, SInt, 16);
class LittleUInt32 __OSEndianSignIntSizeDEF(LittleUInt32, Little, UInt, 32);
class LittleSInt32 __OSEndianSignIntSizeDEF(LittleSInt32, Little, SInt, 32);
class LittleUInt64 __OSEndianSignIntSizeDEF(LittleUInt64, Little, UInt, 64);
class LittleSInt64 __OSEndianSignIntSizeDEF(LittleSInt64, Little, SInt, 64);

#undef __OSEndianSignIntSizeDEF

#endif /* __cplusplus
        */

#endif /* ! _OS_OSENDIANHELPER_H
        */
