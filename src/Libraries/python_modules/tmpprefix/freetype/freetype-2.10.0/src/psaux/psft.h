/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#ifndef PSFT_H_
#define PSFT_H_


#include "pstypes.h"


  /* TODO: disable asserts for now */
#define CF2_NDEBUG


#include FT_SYSTEM_H

#include "psglue.h"
#include FT_INTERNAL_POSTSCRIPT_AUX_H    /* for PS_Decoder */


FT_BEGIN_HEADER


  FT_LOCAL( FT_Error )
  cf2_decoder_parse_charstrings( PS_Decoder*  decoder,
                                 FT_Byte*     charstring_base,
                                 FT_ULong     charstring_len );

  FT_LOCAL( CFF_SubFont )
  cf2_getSubfont( PS_Decoder*  decoder );

  FT_LOCAL( CFF_VStore )
  cf2_getVStore( PS_Decoder*  decoder );

  FT_LOCAL( FT_UInt )
  cf2_getMaxstack( PS_Decoder*  decoder );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
  FT_LOCAL( FT_Error )
  cf2_getNormalizedVector( PS_Decoder*  decoder,
                           CF2_UInt    *len,
                           FT_Fixed*   *vec );
#endif

  FT_LOCAL( CF2_Fixed )
  cf2_getPpemY( PS_Decoder*  decoder );
  FT_LOCAL( CF2_Fixed )
  cf2_getStdVW( PS_Decoder*  decoder );
  FT_LOCAL( CF2_Fixed )
  cf2_getStdHW( PS_Decoder*  decoder );

  FT_LOCAL( void )
  cf2_getBlueMetrics( PS_Decoder*  decoder,
                      CF2_Fixed*   blueScale,
                      CF2_Fixed*   blueShift,
                      CF2_Fixed*   blueFuzz );
  FT_LOCAL( void )
  cf2_getBlueValues( PS_Decoder*  decoder,
                     size_t*      count,
                     FT_Pos*     *data );
  FT_LOCAL( void )
  cf2_getOtherBlues( PS_Decoder*  decoder,
                     size_t*      count,
                     FT_Pos*     *data );
  FT_LOCAL( void )
  cf2_getFamilyBlues( PS_Decoder*  decoder,
                      size_t*      count,
                      FT_Pos*     *data );
  FT_LOCAL( void )
  cf2_getFamilyOtherBlues( PS_Decoder*  decoder,
                           size_t*      count,
                           FT_Pos*     *data );

  FT_LOCAL( CF2_Int )
  cf2_getLanguageGroup( PS_Decoder*  decoder );

  FT_LOCAL( CF2_Int )
  cf2_initGlobalRegionBuffer( PS_Decoder*  decoder,
                              CF2_Int      subrNum,
                              CF2_Buffer   buf );
  FT_LOCAL( FT_Error )
  cf2_getSeacComponent( PS_Decoder*  decoder,
                        CF2_Int      code,
                        CF2_Buffer   buf );
  FT_LOCAL( void )
  cf2_freeSeacComponent( PS_Decoder*  decoder,
                         CF2_Buffer   buf );
  FT_LOCAL( CF2_Int )
  cf2_initLocalRegionBuffer( PS_Decoder*  decoder,
                             CF2_Int      subrNum,
                             CF2_Buffer   buf );

  FT_LOCAL( CF2_Fixed )
  cf2_getDefaultWidthX( PS_Decoder*  decoder );
  FT_LOCAL( CF2_Fixed )
  cf2_getNominalWidthX( PS_Decoder*  decoder );


  FT_LOCAL( FT_Error )
  cf2_getT1SeacComponent( PS_Decoder*  decoder,
                          FT_UInt      glyph_index,
                          CF2_Buffer   buf );
  FT_LOCAL( void )
  cf2_freeT1SeacComponent( PS_Decoder*  decoder,
                           CF2_Buffer   buf );

  /*
   * FreeType client outline
   *
   * process output from the charstring interpreter
   */
  typedef struct  CF2_OutlineRec_
  {
    CF2_OutlineCallbacksRec  root;        /* base class must be first */
    PS_Decoder*              decoder;

  } CF2_OutlineRec, *CF2_Outline;


  FT_LOCAL( void )
  cf2_outline_reset( CF2_Outline  outline );
  FT_LOCAL( void )
  cf2_outline_close( CF2_Outline  outline );


FT_END_HEADER


#endif /* PSFT_H_ */


/* END */
