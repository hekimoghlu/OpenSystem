/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#ifndef __BIGNUM128_H__
#define __BIGNUM128_H__

#include <libkern/OSTypes.h>
#include <stdint.h>

class U128
{
public:
	U128(uint64_t _lo = 0) : lo(_lo), hi(0)				{ };
	U128(uint64_t _hi, uint64_t _lo)	: lo(_lo), hi(_hi)	{ };
	inline bool operator==( const U128 &A ) const	 	{ return ( A.hi == hi ) && ( A.lo == lo ); }
	inline bool operator>( const U128 &A ) const		{ return ( ( A.hi > hi ) || ( ( A.hi == hi ) && ( A.lo > lo ) ) ); }
	inline bool operator<( const U128 &A ) const 		{ return !( ( A.hi > hi ) || ( ( A.hi == hi ) && ( A.lo > lo ) ) ); }

	U128 operator++( int )
	{
		if ( ++lo==0 )
			hi++;

		return *this;
	}
	
	U128 operator--( int )
	{
		if ( 0 == lo-- )
		{
			hi--;
		}
		return *this;
	}

	U128& operator=( const U128 &A )
	{
		hi = A.hi;
		lo = A.lo;
		
		return *this;
	}
	
	U128 operator+( const U128 &A ) const
	{
		U128	result(A.hi + hi, A.lo + lo);
		
		if ( ( result.lo < A.lo ) || ( result.lo < lo ) )
		{
			result.hi++;
		}
		
		return result;
	}
	
	U128& operator+=( const U128 &A )
	{
		U128	result(A.hi + hi, A.lo + lo);
		
		if ( ( result.lo < A.lo ) || ( result.lo < lo ) )
		{
			result.hi++;
		}
		
		*this = result;

		return *this;
	}

	friend U128 operator-( const U128 &A, const U128 &B )		// assumes A >= B
	{
		U128 C = A;

		C.hi -= B.hi;
		C.lo -= B.lo;

		if ( C.lo > A.lo )		 // borrow ?
		{
			C.hi--;
		}

		return C;
	}

	
	friend U128 operator<<( const U128& A, int n )
	{
		U128 res = A;
		
		while ( n-- )
		{
			res.hi <<= 1;
			res.hi |= ( ( res.lo & MSB64 ) ? 1 : 0 );
			res.lo <<= 1;
		}
		
		return res;
	}
	
	friend U128 operator>>( const U128& A, int n )
	{
		U128 res = A;

		while ( n-- )
		{
			res.lo >>= 1;
			res.lo |= ( ( res.hi & 0x1 ) ? MSB64 : 0 );
			res.hi >>= 1;
		}
		
		return res;
	}

public:

#ifdef __BIG_ENDIAN__
	uint64_t		hi;
	uint64_t		lo;
#else
	uint64_t		lo;
	uint64_t		hi;
#endif

private:
	enum { MSB64 = 0x8000000000000000ULL };
};

extern U128 UInt64mult(const uint64_t A, const uint64_t B);

#endif			//__BIGNUM128_H__





