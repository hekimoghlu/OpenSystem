/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
/*
 * Copyright (c) 1992 NeXT Computer, Inc.
 *
 * Byte ordering conversion.
 *
 */

#ifndef	_ARCHITECTURE_BYTE_ORDER_H_
#define _ARCHITECTURE_BYTE_ORDER_H_

/*
 * Please note that the byte ordering functions in this file are deprecated.
 * A replacement API exists in libkern/OSByteOrder.h
 */

#include <libkern/OSByteOrder.h>

typedef unsigned long NXSwappedFloat;
typedef unsigned long long NXSwappedDouble;

static __inline__ __attribute__((deprecated))
unsigned short
NXSwapShort(
    unsigned short inv
)
{
    return (unsigned short)OSSwapInt16((uint16_t)inv);
}

static __inline__ __attribute__((deprecated))
unsigned int
NXSwapInt(
    unsigned int inv
)
{
    return (unsigned int)OSSwapInt32((uint32_t)inv);
}

static __inline__ __attribute__((deprecated))
unsigned long
NXSwapLong(
    unsigned long inv
)
{
    return (unsigned long)OSSwapInt32((uint32_t)inv);
}

static __inline__ __attribute__((deprecated))
unsigned long long
NXSwapLongLong(
    unsigned long long inv
)
{
    return (unsigned long long)OSSwapInt64((uint64_t)inv);
}

static __inline__ __attribute__((deprecated))
NXSwappedFloat
NXConvertHostFloatToSwapped(float x)
{
    union fconv {
        float number;
        NXSwappedFloat sf;
    } u;
    u.number = x;
    return u.sf;
}

static __inline__ __attribute__((deprecated))
float
NXConvertSwappedFloatToHost(NXSwappedFloat x)
{
    union fconv {
        float number;
        NXSwappedFloat sf;
    } u;
    u.sf = x;
    return u.number;
}

static __inline__ __attribute__((deprecated))
NXSwappedDouble
NXConvertHostDoubleToSwapped(double x)
{
    union dconv {
        double number;
        NXSwappedDouble sd;
    } u;
    u.number = x;
    return u.sd;
}

static __inline__ __attribute__((deprecated))
double
NXConvertSwappedDoubleToHost(NXSwappedDouble x)
{
    union dconv {
        double number;
        NXSwappedDouble sd;
    } u;
    u.sd = x;
    return u.number;
}

static __inline__ __attribute__((deprecated))
NXSwappedFloat
NXSwapFloat(NXSwappedFloat x)
{
    return (NXSwappedFloat)OSSwapInt32((uint32_t)x);
}

static __inline__ __attribute__((deprecated))
NXSwappedDouble
NXSwapDouble(NXSwappedDouble x)
{
    return (NXSwappedDouble)OSSwapInt64((uint64_t)x);
}

/*
 * Identify the byte order
 * of the current host.
 */

enum NXByteOrder {
    NX_UnknownByteOrder,
    NX_LittleEndian,
    NX_BigEndian
};

static __inline__
enum NXByteOrder
NXHostByteOrder(void)
{
#if defined(__LITTLE_ENDIAN__)
    return NX_LittleEndian;
#elif defined(__BIG_ENDIAN__)
    return NX_BigEndian;
#else
    return NX_UnknownByteOrder;
#endif
}

static __inline__ __attribute__((deprecated))
unsigned short
NXSwapBigShortToHost(
    unsigned short	x
)
{
    return (unsigned short)OSSwapBigToHostInt16((uint16_t)x);
}

static __inline__ __attribute__((deprecated))
unsigned int
NXSwapBigIntToHost(
    unsigned int	x
)
{
    return (unsigned int)OSSwapBigToHostInt32((uint32_t)x);
}

static __inline__ __attribute__((deprecated))
unsigned long
NXSwapBigLongToHost(
    unsigned long	x
)
{
    return (unsigned long)OSSwapBigToHostInt32((uint32_t)x);
}

static __inline__ __attribute__((deprecated))
unsigned long long
NXSwapBigLongLongToHost(
    unsigned long long	x
)
{
    return (unsigned long long)OSSwapBigToHostInt64((uint64_t)x);
}

static __inline__ __attribute__((deprecated))
double
NXSwapBigDoubleToHost(
    NXSwappedDouble	x
)
{
    return NXConvertSwappedDoubleToHost((NXSwappedDouble)OSSwapBigToHostInt64((uint64_t)x));
}

static __inline__ __attribute__((deprecated))
float
NXSwapBigFloatToHost(
    NXSwappedFloat	x
)
{
    return NXConvertSwappedFloatToHost((NXSwappedFloat)OSSwapBigToHostInt32((uint32_t)x));
}

static __inline__ __attribute__((deprecated))
unsigned short
NXSwapHostShortToBig(
    unsigned short	x
)
{
    return (unsigned short)OSSwapHostToBigInt16((uint16_t)x);
}

static __inline__ __attribute__((deprecated))
unsigned int
NXSwapHostIntToBig(
    unsigned int	x
)
{
    return (unsigned int)OSSwapHostToBigInt32((uint32_t)x);
}

static __inline__ __attribute__((deprecated))
unsigned long
NXSwapHostLongToBig(
    unsigned long	x
)
{
    return (unsigned long)OSSwapHostToBigInt32((uint32_t)x);
}

static __inline__ __attribute__((deprecated))
unsigned long long
NXSwapHostLongLongToBig(
    unsigned long long	x
)
{
    return (unsigned long long)OSSwapHostToBigInt64((uint64_t)x);
}

static __inline__ __attribute__((deprecated))
NXSwappedDouble
NXSwapHostDoubleToBig(
    double	x
)
{
    return (NXSwappedDouble)OSSwapHostToBigInt64((uint64_t)NXConvertHostDoubleToSwapped(x));
}

static __inline__ __attribute__((deprecated))
NXSwappedFloat
NXSwapHostFloatToBig(
    float	x
)
{
    return (NXSwappedFloat)OSSwapHostToBigInt32((uint32_t)NXConvertHostFloatToSwapped(x));
}

static __inline__ __attribute__((deprecated))
unsigned short
NXSwapLittleShortToHost(
    unsigned short	x
)
{
    return (unsigned short)OSSwapLittleToHostInt16((uint16_t)x);
}

static __inline__ __attribute__((deprecated))
unsigned int
NXSwapLittleIntToHost(
    unsigned int	x
)
{
    return (unsigned int)OSSwapLittleToHostInt32((uint32_t)x);
}

static __inline__ __attribute__((deprecated))
unsigned long
NXSwapLittleLongToHost(
    unsigned long	x
)
{
    return (unsigned long)OSSwapLittleToHostInt32((uint32_t)x);
}

static __inline__ __attribute__((deprecated))
unsigned long long
NXSwapLittleLongLongToHost(
    unsigned long long	x
)
{
    return (unsigned long long)OSSwapLittleToHostInt64((uint64_t)x);
}

static __inline__ __attribute__((deprecated))
double
NXSwapLittleDoubleToHost(
    NXSwappedDouble	x
)
{
    return NXConvertSwappedDoubleToHost((NXSwappedDouble)OSSwapLittleToHostInt64((uint64_t)x));
}

static __inline__ __attribute__((deprecated))
float
NXSwapLittleFloatToHost(
    NXSwappedFloat	x
)
{
    return NXConvertSwappedFloatToHost((NXSwappedFloat)OSSwapLittleToHostInt32((uint32_t)x));
}

static __inline__ __attribute__((deprecated))
unsigned short
NXSwapHostShortToLittle(
    unsigned short	x
)
{
    return (unsigned short)OSSwapHostToLittleInt16((uint16_t)x);
}

static __inline__ __attribute__((deprecated))
unsigned int
NXSwapHostIntToLittle(
    unsigned int	x
)
{
    return (unsigned int)OSSwapHostToLittleInt32((uint32_t)x);
}

static __inline__ __attribute__((deprecated))
unsigned long
NXSwapHostLongToLittle(
    unsigned long	x
)
{
    return (unsigned long)OSSwapHostToLittleInt32((uint32_t)x);
}

static __inline__ __attribute__((deprecated))
unsigned long long
NXSwapHostLongLongToLittle(
    unsigned long long	x
)
{
    return (unsigned long long)OSSwapHostToLittleInt64((uint64_t)x);
}

static __inline__ __attribute__((deprecated))
NXSwappedDouble
NXSwapHostDoubleToLittle(
    double	x
)
{
    return (NXSwappedDouble)OSSwapHostToLittleInt64((uint64_t)NXConvertHostDoubleToSwapped(x));
}

static __inline__ __attribute__((deprecated))
NXSwappedFloat
NXSwapHostFloatToLittle(
    float	x
)
{
    return (NXSwappedFloat)OSSwapHostToLittleInt32((uint32_t)NXConvertHostFloatToSwapped(x));
}

#endif	/* _ARCHITECTURE_BYTE_ORDER_H_ */
