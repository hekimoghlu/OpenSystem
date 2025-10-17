/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_TRUETYPE_TAGS_H
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_STREAM_H
#include FT_SERVICE_SFNT_H
#include FT_SERVICE_TRUETYPE_GLYF_H


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Bool )
  FT_Face_CheckTrueTypePatents( FT_Face  face )
  {
    FT_UNUSED( face );

    return FALSE;
  }


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Bool )
  FT_Face_SetUnpatentedHinting( FT_Face  face,
                                FT_Bool  value )
  {
    FT_UNUSED( face );
    FT_UNUSED( value );

    return FALSE;
  }

/* END */
