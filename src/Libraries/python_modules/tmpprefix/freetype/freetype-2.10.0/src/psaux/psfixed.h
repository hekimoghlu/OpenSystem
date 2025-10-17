/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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
#ifndef PSFIXED_H_
#define PSFIXED_H_


FT_BEGIN_HEADER


  /* rasterizer integer and fixed point arithmetic must be 32-bit */

#define   CF2_Fixed  CF2_F16Dot16
  typedef FT_Int32   CF2_Frac;   /* 2.30 fixed point */


#define CF2_FIXED_MAX      ( (CF2_Fixed)0x7FFFFFFFL )
#define CF2_FIXED_MIN      ( (CF2_Fixed)0x80000000L )
#define CF2_FIXED_ONE      ( (CF2_Fixed)0x10000L )
#define CF2_FIXED_EPSILON  ( (CF2_Fixed)0x0001 )

  /* in C 89, left and right shift of negative numbers is  */
  /* implementation specific behaviour in the general case */

#define cf2_intToFixed( i )                                              \
          ( (CF2_Fixed)( (FT_UInt32)(i) << 16 ) )
#define cf2_fixedToInt( x )                                              \
          ( (FT_Short)( ( (FT_UInt32)(x) + 0x8000U ) >> 16 ) )
#define cf2_fixedRound( x )                                              \
          ( (CF2_Fixed)( ( (FT_UInt32)(x) + 0x8000U ) & 0xFFFF0000UL ) )
#define cf2_doubleToFixed( f )                                           \
          ( (CF2_Fixed)( (f) * 65536.0 + 0.5 ) )
#define cf2_fixedAbs( x )                                                \
          ( (x) < 0 ? NEG_INT32( x ) : (x) )
#define cf2_fixedFloor( x )                                              \
          ( (CF2_Fixed)( (FT_UInt32)(x) & 0xFFFF0000UL ) )
#define cf2_fixedFraction( x )                                           \
          ( (x) - cf2_fixedFloor( x ) )
#define cf2_fracToFixed( x )                                             \
          ( (x) < 0 ? -( ( -(x) + 0x2000 ) >> 14 )                       \
                    :  ( (  (x) + 0x2000 ) >> 14 ) )


  /* signed numeric types */
  typedef enum  CF2_NumberType_
  {
    CF2_NumberFixed,    /* 16.16 */
    CF2_NumberFrac,     /*  2.30 */
    CF2_NumberInt       /* 32.0  */

  } CF2_NumberType;


FT_END_HEADER


#endif /* PSFIXED_H_ */


/* END */
