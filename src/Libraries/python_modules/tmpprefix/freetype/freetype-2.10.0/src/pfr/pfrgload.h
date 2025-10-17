/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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
#ifndef PFRGLOAD_H_
#define PFRGLOAD_H_

#include "pfrtypes.h"

FT_BEGIN_HEADER


  FT_LOCAL( void )
  pfr_glyph_init( PFR_Glyph       glyph,
                  FT_GlyphLoader  loader );

  FT_LOCAL( void )
  pfr_glyph_done( PFR_Glyph  glyph );


  FT_LOCAL( FT_Error )
  pfr_glyph_load( PFR_Glyph  glyph,
                  FT_Stream  stream,
                  FT_ULong   gps_offset,
                  FT_ULong   offset,
                  FT_ULong   size );


FT_END_HEADER


#endif /* PFRGLOAD_H_ */


/* END */
