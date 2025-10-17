/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 20, 2023.
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
/****************************************************
   *
   * This file is *not* portable!  You have to adapt
   * its definitions to your platform.
   *
   */

#ifndef FTMISC_H_
#define FTMISC_H_


  /* memset */
#include FT_CONFIG_STANDARD_LIBRARY_H

#define FT_BEGIN_HEADER
#define FT_END_HEADER

#define FT_LOCAL_DEF( x )   static x


  /* from include/freetype/fttypes.h */

  typedef unsigned char  FT_Byte;
  typedef signed int     FT_Int;
  typedef unsigned int   FT_UInt;
  typedef signed long    FT_Long;
  typedef unsigned long  FT_ULong;
  typedef signed long    FT_F26Dot6;
  typedef int            FT_Error;

#define FT_MAKE_TAG( _x1, _x2, _x3, _x4 ) \
          ( ( (FT_ULong)_x1 << 24 ) |     \
            ( (FT_ULong)_x2 << 16 ) |     \
            ( (FT_ULong)_x3 <<  8 ) |     \
              (FT_ULong)_x4         )


  /* from include/freetype/ftsystem.h */

  typedef struct FT_MemoryRec_*  FT_Memory;

  typedef void* (*FT_Alloc_Func)( FT_Memory  memory,
                                  long       size );

  typedef void (*FT_Free_Func)( FT_Memory  memory,
                                void*      block );

  typedef void* (*FT_Realloc_Func)( FT_Memory  memory,
                                    long       cur_size,
                                    long       new_size,
                                    void*      block );

  typedef struct FT_MemoryRec_
  {
    void*            user;

    FT_Alloc_Func    alloc;
    FT_Free_Func     free;
    FT_Realloc_Func  realloc;

  } FT_MemoryRec;


  /* from src/ftcalc.c */

#if ( defined _WIN32 || defined _WIN64 )

  typedef __int64  FT_Int64;

#else

#include "inttypes.h"

  typedef int64_t  FT_Int64;

#endif


  static FT_Long
  FT_MulDiv( FT_Long  a,
             FT_Long  b,
             FT_Long  c )
  {
    FT_Int   s;
    FT_Long  d;


    s = 1;
    if ( a < 0 ) { a = -a; s = -1; }
    if ( b < 0 ) { b = -b; s = -s; }
    if ( c < 0 ) { c = -c; s = -s; }

    d = (FT_Long)( c > 0 ? ( (FT_Int64)a * b + ( c >> 1 ) ) / c
                         : 0x7FFFFFFFL );

    return ( s > 0 ) ? d : -d;
  }


  static FT_Long
  FT_MulDiv_No_Round( FT_Long  a,
                      FT_Long  b,
                      FT_Long  c )
  {
    FT_Int   s;
    FT_Long  d;


    s = 1;
    if ( a < 0 ) { a = -a; s = -1; }
    if ( b < 0 ) { b = -b; s = -s; }
    if ( c < 0 ) { c = -c; s = -s; }

    d = (FT_Long)( c > 0 ? (FT_Int64)a * b / c
                         : 0x7FFFFFFFL );

    return ( s > 0 ) ? d : -d;
  }

#endif /* FTMISC_H_ */


/* END */
