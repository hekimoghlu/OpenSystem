/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 28, 2023.
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
/*=============================================================================
	PCMBlitterLibDispatch.h
	
=============================================================================*/

#include "IOAudioBlitterLibDispatch.h"

#include "IOAudioBlitterLib.h"
#include <xmmintrin.h>
#include <smmintrin.h>

/*
	PCM int<->float library.

	These are the high-level interfaces which dispatch to (often processor-specific) optimized routines.
	Avoid calling the lower-level routines directly; they are subject to renaming etc.
	
	There are two sets of interfaces:
	[1] integer formats are either "native" or "swap"
	[2] integer formats are "BE" or "LE", signifying big or little endian. These are simply macros for the other functions.
	
	All floating point numbers are 32-bit native-endian.
	Supports 16, 24, and 32-bit integers, big and little endian.
	
	32-bit floats and ints must be 4-byte aligned.
	24-bit samples have no alignment requirements.
	16-bit ints must be 2-byte aligned.
	
	On Intel, the haveVector argument is ignored and some implementations assume SSE2.
*/



void IOAF_NativeInt16ToFloat32( const SInt16 *src, Float32 *dest, unsigned int count )
{
	NativeInt16ToFloat32_X86(src, dest, count);
}

void IOAF_SwapInt16ToFloat32( const SInt16 *src, Float32 *dest, unsigned int count )
{
	SwapInt16ToFloat32_X86(src, dest, count);
}

void IOAF_NativeInt24ToFloat32( const UInt8 *src, Float32 *dest, unsigned int count )
{
	NativeInt24ToFloat32_Portable(src, dest, count);
}

void IOAF_SwapInt24ToFloat32( const UInt8 *src, Float32 *dest, unsigned int count )
{
	SwapInt24ToFloat32_Portable(src, dest, count);
}

void IOAF_NativeInt32ToFloat32( const SInt32 *src, Float32 *dest, unsigned int count )
{
	NativeInt32ToFloat32_X86(src, dest, count);
}

void IOAF_SwapInt32ToFloat32( const SInt32 *src, Float32 *dest, unsigned int count )
{
	SwapInt32ToFloat32_X86(src, dest, count);
}

void IOAF_Float32ToNativeInt16( const Float32 *src, SInt16 *dest, unsigned int count )
{
	Float32ToNativeInt16_X86(src, dest, count);
}

void IOAF_Float32ToSwapInt16( const Float32 *src, SInt16 *dest, unsigned int count )
{
	Float32ToSwapInt16_X86(src, dest, count);
}

void IOAF_Float32ToNativeInt24( const Float32 *src, UInt8 *dest, unsigned int count )
{
	Float32ToNativeInt24_X86(src, dest, count);
}

void IOAF_Float32ToSwapInt24( const Float32 *src, UInt8 *dest, unsigned int count )
{
	Float32ToSwapInt24_Portable(src, dest, count);
}

void IOAF_Float32ToNativeInt32( const Float32 *src, SInt32 *dest, unsigned int count )
{
	Float32ToNativeInt32_X86(src, dest, count);
}

void IOAF_Float32ToSwapInt32( const Float32 *src, SInt32 *dest, unsigned int count )
{
	Float32ToSwapInt32_X86(src, dest, count);
}

void IOAF_bcopy_WriteCombine(const void *pSrc, void *pDst, unsigned int count)
{
	unsigned int n;
	
	bool salign = !((uintptr_t)pSrc & 0xF);
	bool dalign = !((uintptr_t)pDst & 0xF);
	unsigned int size4Left = count & 0xF;
	UInt8*	src_data = (UInt8*) pSrc;
	UInt8*	dst_data = (UInt8*) pDst;
	
	count &= ~0xF;
	
	if ( dalign )
	{
		if ( salign )
		{
			// #1 efficient loop - both src/dst are 16-byte aligned
			
			unsigned int size16Left = count & 0x3F;
			count &= ~0x3F;
			
			// First loop operates on 64 byte chunks
			for (n=0; n<count; n+=64)
			{
				__m128i data1, data2, data3, data4;
				
				data1 = _mm_stream_load_si128( (__m128i*) src_data + 0);
				data2 = _mm_stream_load_si128( (__m128i*) src_data + 1);
				data3 = _mm_stream_load_si128( (__m128i*) src_data + 2);
				data4 = _mm_stream_load_si128( (__m128i*) src_data + 3);
				
				_mm_store_si128((__m128i*)dst_data + 0, data1);
				_mm_store_si128((__m128i*)dst_data + 1, data2);
				_mm_store_si128((__m128i*)dst_data + 2, data3);
				_mm_store_si128((__m128i*)dst_data + 3, data4);
				
				src_data += 64;
				dst_data += 64;
			}
			
			// Second loop works on 16-byte chunks
			for (n=0; n<size16Left; n+=16)
			{
				__m128i data1;
				data1 = _mm_stream_load_si128( (__m128i*) src_data);
				_mm_store_si128((__m128i*)dst_data, data1);
				
				src_data += 16;
				dst_data += 16;				
			}
		}
		else
		{
			// #3 efficient loop - src unaligned (no streaming reads). dst aligned
			for (n=0; n<count; n+=16)
			{
				__m128i data128 = _mm_loadu_si128( (__m128i*) src_data);
				_mm_store_si128((__m128i*)dst_data, data128);
				
				src_data += 16;
				dst_data += 16;
			}			
		}
		
	}
	else
	{
		if ( salign)
		{
			// #2 efficient loop - src aligned, dst not aligned
			for (n=0; n<count; n+=16)
			{
				__m128i data128 = _mm_stream_load_si128( (__m128i*) src_data);
				_mm_storeu_si128((__m128i*)dst_data, data128);
				
				src_data += 16;
				dst_data += 16;
			}
		}
		else
		{
			// #4 efficient loop - src unaligned (no streaming reads). dst unaligned
			for (n=0; n<count; n+=16)
			{
				__m128i data128 = _mm_loadu_si128( (__m128i*) src_data);
				_mm_storeu_si128((__m128i*)dst_data, data128);
				
				src_data += 16;
				dst_data += 16;
			}			
		}
	}
	
	// Last loop works on any remaining data not transfered
	for (n=0; n < size4Left; n++)
		*(((char*)dst_data++)) = *((char*)src_data++);
	
	_mm_mfence();
}
