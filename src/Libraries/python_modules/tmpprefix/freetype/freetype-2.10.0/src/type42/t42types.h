/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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
#ifndef T42TYPES_H_
#define T42TYPES_H_


#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_TYPE1_TABLES_H
#include FT_INTERNAL_TYPE1_TYPES_H
#include FT_INTERNAL_POSTSCRIPT_HINTS_H


FT_BEGIN_HEADER


  typedef struct  T42_FaceRec_
  {
    FT_FaceRec      root;
    T1_FontRec      type1;
    const void*     psnames;
    const void*     psaux;
#if 0
    const void*     afm_data;
#endif
    FT_Byte*        ttf_data;
    FT_Long         ttf_size;
    FT_Face         ttf_face;
    FT_CharMapRec   charmaprecs[2];
    FT_CharMap      charmaps[2];
    PS_UnicodesRec  unicode_map;

  } T42_FaceRec, *T42_Face;


FT_END_HEADER

#endif /* T42TYPES_H_ */


/* END */
