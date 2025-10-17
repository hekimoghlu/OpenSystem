/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
#ifndef T1AFM_H_
#define T1AFM_H_

#include <ft2build.h>
#include "t1objs.h"
#include FT_INTERNAL_TYPE1_TYPES_H

FT_BEGIN_HEADER


  FT_LOCAL( FT_Error )
  T1_Read_Metrics( FT_Face    face,
                   FT_Stream  stream );

  FT_LOCAL( void )
  T1_Done_Metrics( FT_Memory     memory,
                   AFM_FontInfo  fi );

  FT_LOCAL( void )
  T1_Get_Kerning( AFM_FontInfo  fi,
                  FT_UInt       glyph1,
                  FT_UInt       glyph2,
                  FT_Vector*    kerning );

  FT_LOCAL( FT_Error )
  T1_Get_Track_Kerning( FT_Face    face,
                        FT_Fixed   ptsize,
                        FT_Int     degree,
                        FT_Fixed*  kerning );

FT_END_HEADER

#endif /* T1AFM_H_ */


/* END */
