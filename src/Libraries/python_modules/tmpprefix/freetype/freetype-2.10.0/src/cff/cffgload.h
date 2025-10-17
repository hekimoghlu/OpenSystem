/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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
#ifndef CFFGLOAD_H_
#define CFFGLOAD_H_


#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_INTERNAL_CFF_OBJECTS_TYPES_H


FT_BEGIN_HEADER

  FT_LOCAL( FT_Error )
  cff_get_glyph_data( TT_Face    face,
                      FT_UInt    glyph_index,
                      FT_Byte**  pointer,
                      FT_ULong*  length );
  FT_LOCAL( void )
  cff_free_glyph_data( TT_Face    face,
                       FT_Byte**  pointer,
                       FT_ULong   length );


#if 0  /* unused until we support pure CFF fonts */

  /* Compute the maximum advance width of a font through quick parsing */
  FT_LOCAL( FT_Error )
  cff_compute_max_advance( TT_Face  face,
                           FT_Int*  max_advance );

#endif /* 0 */


  FT_LOCAL( FT_Error )
  cff_slot_load( CFF_GlyphSlot  glyph,
                 CFF_Size       size,
                 FT_UInt        glyph_index,
                 FT_Int32       load_flags );


FT_END_HEADER

#endif /* CFFGLOAD_H_ */


/* END */
