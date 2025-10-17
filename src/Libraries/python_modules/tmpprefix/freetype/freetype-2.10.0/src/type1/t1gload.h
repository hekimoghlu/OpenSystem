/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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
#ifndef T1GLOAD_H_
#define T1GLOAD_H_


#include <ft2build.h>
#include "t1objs.h"


FT_BEGIN_HEADER


  FT_LOCAL( FT_Error )
  T1_Compute_Max_Advance( T1_Face  face,
                          FT_Pos*  max_advance );

  FT_LOCAL( FT_Error )
  T1_Get_Advances( FT_Face    face,
                   FT_UInt    first,
                   FT_UInt    count,
                   FT_Int32   load_flags,
                   FT_Fixed*  advances );

  FT_LOCAL( FT_Error )
  T1_Load_Glyph( FT_GlyphSlot  glyph,
                 FT_Size       size,
                 FT_UInt       glyph_index,
                 FT_Int32      load_flags );


FT_END_HEADER

#endif /* T1GLOAD_H_ */


/* END */
