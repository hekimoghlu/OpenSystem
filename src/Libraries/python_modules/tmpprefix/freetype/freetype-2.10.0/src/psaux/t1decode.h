/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 1, 2024.
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
#ifndef T1DECODE_H_
#define T1DECODE_H_


#include <ft2build.h>
#include FT_INTERNAL_POSTSCRIPT_AUX_H
#include FT_INTERNAL_TYPE1_TYPES_H


FT_BEGIN_HEADER


  FT_CALLBACK_TABLE
  const T1_Decoder_FuncsRec  t1_decoder_funcs;

  FT_LOCAL( FT_Int )
  t1_lookup_glyph_by_stdcharcode_ps( PS_Decoder*  decoder,
                                     FT_Int       charcode );

#ifdef T1_CONFIG_OPTION_OLD_ENGINE
  FT_LOCAL( FT_Error )
  t1_decoder_parse_glyph( T1_Decoder  decoder,
                          FT_UInt     glyph_index );

  FT_LOCAL( FT_Error )
  t1_decoder_parse_charstrings( T1_Decoder  decoder,
                                FT_Byte*    base,
                                FT_UInt     len );
#else
  FT_LOCAL( FT_Error )
  t1_decoder_parse_metrics( T1_Decoder  decoder,
                            FT_Byte*    charstring_base,
                            FT_UInt     charstring_len );
#endif

  FT_LOCAL( FT_Error )
  t1_decoder_init( T1_Decoder           decoder,
                   FT_Face              face,
                   FT_Size              size,
                   FT_GlyphSlot         slot,
                   FT_Byte**            glyph_names,
                   PS_Blend             blend,
                   FT_Bool              hinting,
                   FT_Render_Mode       hint_mode,
                   T1_Decoder_Callback  parse_glyph );

  FT_LOCAL( void )
  t1_decoder_done( T1_Decoder  decoder );


FT_END_HEADER

#endif /* T1DECODE_H_ */


/* END */
