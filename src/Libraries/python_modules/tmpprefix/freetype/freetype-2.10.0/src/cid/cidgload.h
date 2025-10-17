/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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
#ifndef CIDGLOAD_H_
#define CIDGLOAD_H_


#include <ft2build.h>
#include "cidobjs.h"


FT_BEGIN_HEADER


#if 0

  /* Compute the maximum advance width of a font through quick parsing */
  FT_LOCAL( FT_Error )
  cid_face_compute_max_advance( CID_Face  face,
                                FT_Int*   max_advance );

#endif /* 0 */

  FT_LOCAL( FT_Error )
  cid_slot_load_glyph( FT_GlyphSlot  glyph,         /* CID_Glyph_Slot */
                       FT_Size       size,          /* CID_Size       */
                       FT_UInt       glyph_index,
                       FT_Int32      load_flags );


FT_END_HEADER

#endif /* CIDGLOAD_H_ */


/* END */
