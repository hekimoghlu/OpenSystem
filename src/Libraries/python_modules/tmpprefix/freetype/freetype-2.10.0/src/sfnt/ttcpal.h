/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 31, 2025.
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
#ifndef __TTCPAL_H__
#define __TTCPAL_H__


#include <ft2build.h>
#include "ttload.h"


FT_BEGIN_HEADER


  FT_LOCAL( FT_Error )
  tt_face_load_cpal( TT_Face    face,
                     FT_Stream  stream );

  FT_LOCAL( void )
  tt_face_free_cpal( TT_Face  face );

  FT_LOCAL( FT_Error )
  tt_face_palette_set( TT_Face  face,
                       FT_UInt  palette_index );


FT_END_HEADER


#endif /* __TTCPAL_H__ */

/* END */
