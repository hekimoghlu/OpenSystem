/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
#ifndef PSINTRP_H_
#define PSINTRP_H_


#include "psft.h"
#include "pshints.h"


FT_BEGIN_HEADER


  FT_LOCAL( void )
  cf2_hintmask_init( CF2_HintMask  hintmask,
                     FT_Error*     error );
  FT_LOCAL( FT_Bool )
  cf2_hintmask_isValid( const CF2_HintMask  hintmask );
  FT_LOCAL( FT_Bool )
  cf2_hintmask_isNew( const CF2_HintMask  hintmask );
  FT_LOCAL( void )
  cf2_hintmask_setNew( CF2_HintMask  hintmask,
                       FT_Bool       val );
  FT_LOCAL( FT_Byte* )
  cf2_hintmask_getMaskPtr( CF2_HintMask  hintmask );
  FT_LOCAL( void )
  cf2_hintmask_setAll( CF2_HintMask  hintmask,
                       size_t        bitCount );

  FT_LOCAL( void )
  cf2_interpT2CharString( CF2_Font              font,
                          CF2_Buffer            charstring,
                          CF2_OutlineCallbacks  callbacks,
                          const FT_Vector*      translation,
                          FT_Bool               doingSeac,
                          CF2_Fixed             curX,
                          CF2_Fixed             curY,
                          CF2_Fixed*            width );


FT_END_HEADER


#endif /* PSINTRP_H_ */


/* END */
