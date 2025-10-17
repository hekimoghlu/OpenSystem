/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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
#ifndef _IOFIXED64_H
#define _IOFIXED64_H

#include <IOKit/IOTypes.h>
#include <machine/limits.h>

#ifndef INT_MAX
#define	INT_MAX	2147483647L	/* max signed int */
#endif

#ifndef INT_MIN
#define	INT_MIN	(-2147483647L-1) /* min signed int */
#endif

//===========================================================================
class IOFixed64 {
public:    
    IOFixed64() {
        value = 0;
    }
    
    SInt64 as64() const {
        return value / 65536LL;
    }
    
    SInt64 asFixed64() const {
        return value;
    }
    
    SInt32 asFixed24x8() const {
        return (SInt32)(value / 256LL);
    }
    
    UInt16 fraction() const {
        UInt64 result;
        if( value < 0LL )
            result = value | ~0xffffLL;
        else
            result = value & 0xffffLL;
        return result;
    }
    
    SInt32 as32() const {
        SInt64 result = as64();
        
        if (result > INT_MAX)
            return INT_MAX;
        
        if (result < INT_MIN)
            return INT_MIN;
        
        return (SInt32)result;
    }
    
    IOFixed asFixed() const {
        if (value > INT_MAX)
            return INT_MAX;
        
        if (value < INT_MIN)
            return INT_MIN;
        
        return IOFixed(value);
    }
    
    IOFixed64& fromIntFloor(SInt64 x) {
        value = (x * 65536LL);
        return *this;
    }
    
    static IOFixed64 withIntFloor(SInt64 x) {
        IOFixed64 result;
        return result.fromIntFloor(x);
    }
    
    IOFixed64& fromIntCeiling(SInt64 x) {
        value = (x * 65536LL) + 65535LL;
        return *this;
    }
    
    IOFixed64& fromFixed(IOFixed x) {
        value = x;
        return *this;
    }
    
    static IOFixed64 withFixed(IOFixed x) {
        IOFixed64 result;
        return result.fromFixed(x);
    }
    
    IOFixed64& fromFixed64(SInt64 x) {
        value = x;
        return *this;
    }
    
    IOFixed64& fromFixed24x8(SInt32 x) {
        value = x * 256LL;
        return *this;
    }
    
    IOFixed64& operator+=(IOFixed64 x) {
        value += x.value;
        return *this;
    }
    
    IOFixed64& operator-=(IOFixed64 x) {
        value -= x.value;
        return *this;
    }
    
    IOFixed64& operator*=(IOFixed64 x) {
        value *= x.value;
        value /= 65536LL;
        return *this;
    }
    
    IOFixed64& operator/=(IOFixed64 x) {
        value *= 65536LL;
        value /= x.value;
        return *this;
    }
    
    IOFixed64& operator+=(SInt64 x) {
        value += x * 65536LL;
        return *this;
    }
    
    IOFixed64& operator-=(SInt64 x) {
        value -= x * 65536LL;
        return *this;
    }
    
    IOFixed64& operator*=(SInt64 x) {
        value *= x;
        return *this;
    }
    
    IOFixed64& operator/=(SInt64 x) {
        value /= x;
        return *this;
    }
    
    operator const bool() {
        return (value != 0);
    }
    
#define BOOL_OPERATOR(X) \
bool operator X (const IOFixed64 b) const { return value X b.value; }; \
bool operator X (const SInt64 b) const { return value X (b * 65536LL); };
    
    BOOL_OPERATOR(>)
    BOOL_OPERATOR(>=)
    BOOL_OPERATOR(<)
    BOOL_OPERATOR(<=)
    BOOL_OPERATOR(==)
    BOOL_OPERATOR(!=)
    
#undef BOOL_OPERATOR

private:
    SInt64 value;
};

//===========================================================================
IOFixed64 operator* (const IOFixed64 a, const IOFixed64 b);
IOFixed64 operator* (const IOFixed64 b, const SInt64 a);
IOFixed64 operator/ (const IOFixed64 a, const IOFixed64 b);
IOFixed64 operator/ (const IOFixed64 a, const SInt64 b);
IOFixed64 operator+ (const IOFixed64 a, const IOFixed64 b);
IOFixed64 operator+ (const IOFixed64 a, const SInt64 b);
IOFixed64 operator- (const IOFixed64 a, const IOFixed64 b);
IOFixed64 operator- (const IOFixed64 a, const SInt64 b);
IOFixed64 exponent(const IOFixed64 original, const UInt8 power);
UInt32 llsqrt(UInt64 x);
UInt16 lsqrt(UInt32 x);
IOFixed64 IOQuarticFunction( const IOFixed64 x, const IOFixed64 *gains );
IOFixed64 IOQuarticDerivative( const IOFixed64 x, const IOFixed64 *gains );

//===========================================================================
#endif // _IOFIXED64_H
