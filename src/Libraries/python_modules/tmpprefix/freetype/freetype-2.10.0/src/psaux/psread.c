/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 18, 2022.
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
#include "psft.h"
#include FT_INTERNAL_DEBUG_H

#include "psglue.h"

#include "pserror.h"


  /* Define CF2_IO_FAIL as 1 to enable random errors and random */
  /* value errors in I/O.                                       */
#define CF2_IO_FAIL  0


#if CF2_IO_FAIL

  /* set the .00 value to a nonzero probability */
  static int
  randomError2( void )
  {
    /* for region buffer ReadByte (interp) function */
    return (double)rand() / RAND_MAX < .00;
  }

  /* set the .00 value to a nonzero probability */
  static CF2_Int
  randomValue()
  {
    return (double)rand() / RAND_MAX < .00 ? rand() : 0;
  }

#endif /* CF2_IO_FAIL */


  /* Region Buffer                                      */
  /*                                                    */
  /* Can be constructed from a copied buffer managed by */
  /* `FCM_getDatablock'.                                */
  /* Reads bytes with check for end of buffer.          */

  /* reading past the end of the buffer sets error and returns zero */
  FT_LOCAL_DEF( CF2_Int )
  cf2_buf_readByte( CF2_Buffer  buf )
  {
    if ( buf->ptr < buf->end )
    {
#if CF2_IO_FAIL
      if ( randomError2() )
      {
        CF2_SET_ERROR( buf->error, Invalid_Stream_Operation );
        return 0;
      }

      return *(buf->ptr)++ + randomValue();
#else
      return *(buf->ptr)++;
#endif
    }
    else
    {
      CF2_SET_ERROR( buf->error, Invalid_Stream_Operation );
      return 0;
    }
  }


  /* note: end condition can occur without error */
  FT_LOCAL_DEF( FT_Bool )
  cf2_buf_isEnd( CF2_Buffer  buf )
  {
    return FT_BOOL( buf->ptr >= buf->end );
  }


/* END */
