/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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
#ifndef TTKERN_H_
#define TTKERN_H_


#include <ft2build.h>
#include FT_INTERNAL_STREAM_H
#include FT_INTERNAL_TRUETYPE_TYPES_H


FT_BEGIN_HEADER


  FT_LOCAL( FT_Error  )
  tt_face_load_kern( TT_Face    face,
                     FT_Stream  stream );

  FT_LOCAL( void )
  tt_face_done_kern( TT_Face  face );

  FT_LOCAL( FT_Int )
  tt_face_get_kerning( TT_Face     face,
                       FT_UInt     left_glyph,
                       FT_UInt     right_glyph );

#define TT_FACE_HAS_KERNING( face )  ( (face)->kern_avail_bits != 0 )


FT_END_HEADER

#endif /* TTKERN_H_ */


/* END */
