/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
	PCMBlitterLib.cpp
	
=============================================================================*/

#include "IOAudioBlitterLib.h"


/*
	This file contains portable int<->float blitters.
*/

// ____________________________________________________________________________
//
void	Float32ToSwapInt24_Portable(const Float32 *src, UInt8 *vdest, unsigned int nSamples)
{
	UInt32 *dest = (UInt32 *)vdest;
	double maxInt32 = 2147483648.0;	// 1 << 31
	double round = 128.0;
	double max32 = maxInt32 - 1.0 - round;
	double min32 = -2147483648.0;
	int shift = 8, count;
	
	SET_ROUNDMODE

	count = nSamples >> 2;
	while (count--) {
		double f1 = src[0] * maxInt32 + round;
		double f2 = src[1] * maxInt32 + round;
		double f3 = src[2] * maxInt32 + round;
		double f4 = src[3] * maxInt32 + round;
		SInt32 i1 = FloatToInt(f1, min32, max32) >> shift;
		SInt32 i2 = FloatToInt(f2, min32, max32) >> shift;
		SInt32 i3 = FloatToInt(f3, min32, max32) >> shift;
		SInt32 i4 = FloatToInt(f4, min32, max32) >> shift;
#if TARGET_RT_BIG_ENDIAN
		// a3 a2 a1 b3
		dest[0] = ((i1 & 0x000000FF) << 24)
				| ((i1 & 0x0000FF00) << 8)
				| ((i1 & 0x00FF0000) >> 8)
				|  (i2 & 0x000000FF);
		// b2 b1 c3 c2
		dest[1] = ((i2 & 0x0000FF00) << 16)
				|  (i2 & 0x00FF0000)
				| ((i3 & 0x000000FF) << 8)
				| ((i3 & 0x0000FF00) >> 8);
		// c1 d3 d2 d1
		dest[2] = ((i3 & 0x00FF0000) << 8)
				| ((i4 & 0x000000FF) << 16)
				|  (i4 & 0x0000FF00)
				| ((i4 & 0x00FF0000) >> 16);
#else
		// memory: a1 a2 a3 b1	register: b1 a3 a2 a1
		dest[0] = ((i2 & 0x00FF0000) << 8)
				| ((i1 & 0x000000FF) << 16)
				|  (i1 & 0x0000FF00)
				| ((i1 & 0x00FF0000) >> 16);
		// memory: b2 b3 c1 c2	register: c2 c1 b3 b2
		dest[1] = ((i3 & 0x0000FF00) << 16)
				|  (i3 & 0x00FF0000)
				| ((i2 & 0x000000FF) << 8)
				| ((i2 & 0x0000FF00) >> 8);
		// memory: c3 d1 d2 d3	register: d3 d2 d1 c3
		dest[2] = ((i4 & 0x000000FF) << 24)
				| ((i4 & 0x0000FF00) << 8)
				| ((i4 & 0x00FF0000) >> 8)
				|  (i3 & 0x000000FF);
#endif
		src += 4;
		dest += 3;
	}
	UInt8 *p = (UInt8 *)dest;
	count = nSamples & 3;
	while (count--) {
		double f1 = *src++ * maxInt32 + round;
		SInt32 i1 = FloatToInt(f1, min32, max32) >> shift;
#if TARGET_RT_BIG_ENDIAN
		p[0] = i1;
		p[1] = i1 >> 8;
		p[2] = i1 >> 16;
#else
		p[0] = UInt8(i1 >> 16);
		p[1] = UInt8(i1 >> 8);
		p[2] = UInt8(i1);
#endif
		p += 3;
	}
	RESTORE_ROUNDMODE
}

// ____________________________________________________________________________
//
void	NativeInt24ToFloat32_Portable( const UInt8 *vsrc, Float32 *dest, unsigned int count )
{
	const UInt32 *src = (const UInt32 *)vsrc;
	Float32 scale = (1. / 2147483648.0);
	int nSamples4 = count >> 2;
	
	while (nSamples4--) {
		SInt32 lv1 = src[0];	// BE: a1 a2 a3 b1		LE memory: a3 a2 a1 b3	register: b3 a1 a2 a3
		SInt32 lv2 = src[1];	// BE: b2 b3 c1 c2		LE memory: b2 b1 c3 c2	register: c2 c3 b1 b2
		SInt32 lv3 = src[2];	// BE: c3 d1 d2 d3		LE memory: c1 d3 d2 d1	register: d1 d2 d3 c1
		SInt32 lv4;
		
//			printf("%08X %08X %08X => ", lv1, lv2, lv3);

#if TARGET_RT_BIG_ENDIAN
		lv4 = lv3 << 8;
		lv3 = (lv2 << 16) | ((lv3 & 0xFF000000) >> 16);
		lv2 = (lv1 << 24) | ((lv2 & 0xFFFF0000) >> 8);
		lv1 &= 0xFFFFFF00;
#else
		lv4 = lv3 & 0xFFFFFF00;
		lv3 = (lv3 << 24) | ((lv2 & 0xFFFF0000) >> 8);
		lv2 = (lv2 << 16) | ((lv1 & 0xFF000000) >> 16);
		lv1 = (lv1 << 8);
#endif
//			printf("%08X %08X %08X %08X\n", lv1, lv2, lv3, lv4);

		dest[0] = lv1 * scale;
		dest[1] = lv2 * scale;
		dest[2] = lv3 * scale;
		dest[3] = lv4 * scale;
		
		src += 3;
		dest += 4;
	}
	int nSamples = count & 3;
	UInt8 *p = (UInt8 *)src;
	while (nSamples--) {
#if TARGET_RT_BIG_ENDIAN
		SInt32 lv = (p[0] << 16) | (p[1] << 8) | p[2];
#else
		SInt32 lv = p[0] | (p[1] << 8) | (p[2] << 16);
#endif
		lv <<= 8;
		p += 3;
		*dest++ = lv * scale;
	}
}

// ____________________________________________________________________________
//
void	SwapInt24ToFloat32_Portable( const UInt8 *vsrc, Float32 *dest, unsigned int count )
{
	const UInt32 *src = (const UInt32 *)vsrc;
	Float32 scale = (1. / 2147483648.0);
	int nSamples4 = count >> 2;
	
	while (nSamples4--) {
		SInt32 lv1 = src[0];	// BE: a3 a2 a1 b3		LE memory: a1 a2 a3 b1	register: b1 a3 a2 a1
		SInt32 lv2 = src[1];	// BE: b2 b1 c3 c2		LE memory: b2 b3 c1 c2	register: c2 c1 b3 b2
		SInt32 lv3 = src[2];	// BE: c1 d3 d2 d1		LE memory: c3 d1 d2 d3	register: d3 d2 d1 c3
		SInt32 lv4;

//			printf("%08X %08X %08X => ", lv1, lv2, lv3);
#if TARGET_RT_BIG_ENDIAN
		lv4 = 	   (lv3 << 24)
				| ((lv3 & 0x0000FF00) << 8)
				| ((lv3 & 0x00FF0000) >> 8);
		lv3 = 	   (lv3 & 0xFF000000)
				| ((lv2 & 0x000000FF) << 16)
				| ((lv2 & 0x0000FF00));
		lv2 =	  ((lv2 & 0x00FF0000) << 8)
				| ((lv2 & 0xFF000000) >> 8)
				| ((lv1 & 0x000000FF) << 8);
		lv1 = 	  ((lv1 & 0x0000FF00) << 16)
				|  (lv1 & 0x00FF0000)
				| ((lv1 & 0xFF000000) >> 16);
#else
		lv4 = 	  ((lv3 & 0x0000FF00) << 16)
				|  (lv3 & 0x00FF0000)
				| ((lv3 & 0xFF000000) >> 16);
		lv3 = 	  ((lv2 & 0x00FF0000) << 8)
				| ((lv2 & 0xFF000000) >> 8)
				| ((lv3 & 0x000000FF) << 8);
		lv2 =     (lv1 & 0xFF000000)
				| ((lv2 & 0x000000FF) << 16)
				|  (lv2 & 0x0000FF00);
		lv1 = 	   (lv1 << 24)
				| ((lv1 & 0x0000FF00) << 8)
				| ((lv1 & 0x00FF0000) >> 8);
#endif
//			printf("%08X %08X %08X %08X\n", lv1, lv2, lv3, lv4);

		dest[0] = lv1 * scale;
		dest[1] = lv2 * scale;
		dest[2] = lv3 * scale;
		dest[3] = lv4 * scale;
		
		src += 3;
		dest += 4;
	}
	int nSamples = count & 3;
	UInt8 *p = (UInt8 *)src;
	while (nSamples--) {
#if TARGET_RT_LITTLE_ENDIAN
		SInt32 lv = (p[0] << 16) | (p[1] << 8) | p[2];
#else
		SInt32 lv = p[0] | (p[1] << 8) | (p[2] << 16);
#endif
		lv <<= 8;
		p += 3;
		*dest++ = lv * scale;
	}
}


