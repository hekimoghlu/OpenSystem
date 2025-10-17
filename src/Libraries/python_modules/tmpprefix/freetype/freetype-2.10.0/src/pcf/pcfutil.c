/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 27, 2022.
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
/* $XFree86: xc/lib/font/util/utilbitmap.c,v 1.3 1999/08/22 08:58:58 dawes Exp $ */

/*
 * Author:  Keith Packard, MIT X Consortium
 */

/* Modified for use with FreeType */


#include <ft2build.h>
#include "pcfutil.h"


  /*
   * Invert bit order within each BYTE of an array.
   */

  FT_LOCAL_DEF( void )
  BitOrderInvert( unsigned char*  buf,
                  size_t          nbytes )
  {
    for ( ; nbytes > 0; nbytes--, buf++ )
    {
      unsigned int  val = *buf;


      val = ( ( val >> 1 ) & 0x55 ) | ( ( val << 1 ) & 0xAA );
      val = ( ( val >> 2 ) & 0x33 ) | ( ( val << 2 ) & 0xCC );
      val = ( ( val >> 4 ) & 0x0F ) | ( ( val << 4 ) & 0xF0 );

      *buf = (unsigned char)val;
    }
  }


  /*
   * Invert byte order within each 16-bits of an array.
   */

  FT_LOCAL_DEF( void )
  TwoByteSwap( unsigned char*  buf,
               size_t          nbytes )
  {
    for ( ; nbytes >= 2; nbytes -= 2, buf += 2 )
    {
      unsigned char  c;


      c      = buf[0];
      buf[0] = buf[1];
      buf[1] = c;
    }
  }

  /*
   * Invert byte order within each 32-bits of an array.
   */

  FT_LOCAL_DEF( void )
  FourByteSwap( unsigned char*  buf,
                size_t          nbytes )
  {
    for ( ; nbytes >= 4; nbytes -= 4, buf += 4 )
    {
      unsigned char  c;


      c      = buf[0];
      buf[0] = buf[3];
      buf[3] = c;

      c      = buf[1];
      buf[1] = buf[2];
      buf[2] = c;
    }
  }


/* END */
