/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
#include "IOFixed64.h"

IOFixed64 operator* (const IOFixed64 a, const IOFixed64 b)
{
    IOFixed64 result(a);
    result *= b;
    return result;
}

IOFixed64 operator* (const IOFixed64 a, const SInt64 b)
{
    IOFixed64 result(a);
    result *= b;
    return result;
}

IOFixed64 operator/ (const IOFixed64 a, const IOFixed64 b)
{
    IOFixed64 result(a);
    result /= b;
    return result;
}

IOFixed64 operator/ (const IOFixed64 a, const SInt64 b)
{
    IOFixed64 result(a);
    result /= b;
    return result;
}

IOFixed64 operator+ (const IOFixed64 a, const IOFixed64 b)
{
    IOFixed64 result(a);
    result += b;
    return result;
}

IOFixed64 operator+ (const IOFixed64 a, const SInt64 b)
{
    IOFixed64 result(a);
    result += b;
    return result;
}

IOFixed64 operator- (const IOFixed64 a, const IOFixed64 b)
{
    IOFixed64 result(a);
    result -= b;
    return result;
}

IOFixed64 operator- (const IOFixed64 a, const SInt64 b)
{
    IOFixed64 result(a);
    result -= b;
    return result;
}

IOFixed64 exponent(const IOFixed64 original, const UInt8 power) 
{
    IOFixed64 result;
    if (power) {
        int i;
        result = original;
        for (i = 1; i < power; i++) {
            result *= original;
        }
    }
    return result;
}

UInt32 llsqrt(UInt64 x)
{
    UInt64 rem = 0;
    UInt64 root = 0;
    int i;
	
    for (i = 0; i < 32; i++) {
        root <<= 1;
        rem = ((rem << 2) + (x >> 62));
        x <<= 2;
		
        root++;
		
        if (root <= rem) {
            rem -=  root;
            root++;
        } else {
            root--;
        }
    }
	
    return(UInt32)(root >> 1);
}

UInt16 lsqrt(UInt32 x)
{
    UInt32 rem = 0;
    UInt32 root = 0;
    int i;
	
    for (i = 0; i < 16; i++) {
        root <<= 1;
        rem = ((rem << 2) + (x >> 30));
        x <<= 2;
		
        root++;
		
        if (root <= rem) {
            rem -=  root;
            root++;
        } else {
            root--;
        }
    }
	
    return(UInt16)(root >> 1);
}

IOFixed64 IOQuarticFunction( const IOFixed64 x, const IOFixed64 *gains )
{
	// Computes hyper-cubic polynomial with 0-intercept: f(x) = m1*x + m2^2 * x^2 + m3^3 * x^3 + m4^4 * x^4 
	IOFixed64 function_at_x = x * gains[0] + exponent(x * gains[1], 2);
    
	// -- Because of IOFixed overhead, don't bother computing higher expansions unless their gain coefficients are non-zero:
	if( gains[2] != 0LL )
		function_at_x += exponent(x * gains[2], 3);
    
	if( gains[3] != 0LL )
		function_at_x += exponent(x * gains[3], 4);
	
	return function_at_x;
}

IOFixed64 IOQuarticDerivative( const IOFixed64 x, const IOFixed64 *gains )
{
	// For hyper-cubic polynomial with 0-intercept: f(x) = m1*x + m2^2 * x^2 + m3^3 * x^3 + m4^4 * x^4 
	// This function evaluates the derivative: f'(x) = m1 + 2 * x * m2^2 + 3 * x^2 * m3^3 + 4 * x^3 * m4^4
	IOFixed64 derivative_at_x = gains[0] + x * exponent(gains[1], 2) * 2LL;
	
	// -- Because of IOFixed overhead, don't bother computing higher expansions unless their gain coefficients are non-zero:
	if( gains[2] != 0LL )
		derivative_at_x += exponent(x, 2) * exponent(gains[2], 3) * 3LL;	
	
	if( gains[3] != 0LL )
		derivative_at_x += exponent(x, 3) * exponent(gains[3], 4) * 4LL;
    
	return derivative_at_x;
}

