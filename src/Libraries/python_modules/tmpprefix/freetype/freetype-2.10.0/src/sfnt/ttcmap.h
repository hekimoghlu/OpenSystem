/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
#ifndef TTCMAP_H_
#define TTCMAP_H_


#include <ft2build.h>
#include FT_INTERNAL_TRUETYPE_TYPES_H
#include FT_INTERNAL_VALIDATE_H
#include FT_SERVICE_TT_CMAP_H

FT_BEGIN_HEADER


#define TT_CMAP_FLAG_UNSORTED     1
#define TT_CMAP_FLAG_OVERLAPPING  2

  typedef struct  TT_CMapRec_
  {
    FT_CMapRec  cmap;
    FT_Byte*    data;           /* pointer to in-memory cmap table */
    FT_Int      flags;          /* for format 4 only               */

  } TT_CMapRec, *TT_CMap;

  typedef const struct TT_CMap_ClassRec_*  TT_CMap_Class;


  typedef FT_Error
  (*TT_CMap_ValidateFunc)( FT_Byte*      data,
                           FT_Validator  valid );

  typedef struct  TT_CMap_ClassRec_
  {
    FT_CMap_ClassRec      clazz;
    FT_UInt               format;
    TT_CMap_ValidateFunc  validate;
    TT_CMap_Info_GetFunc  get_cmap_info;

  } TT_CMap_ClassRec;


#define FT_DEFINE_TT_CMAP( class_,             \
                           size_,              \
                           init_,              \
                           done_,              \
                           char_index_,        \
                           char_next_,         \
                           char_var_index_,    \
                           char_var_default_,  \
                           variant_list_,      \
                           charvariant_list_,  \
                           variantchar_list_,  \
                           format_,            \
                           validate_,          \
                           get_cmap_info_ )    \
  FT_CALLBACK_TABLE_DEF                        \
  const TT_CMap_ClassRec  class_ =             \
  {                                            \
    { size_,                                   \
      init_,                                   \
      done_,                                   \
      char_index_,                             \
      char_next_,                              \
      char_var_index_,                         \
      char_var_default_,                       \
      variant_list_,                           \
      charvariant_list_,                       \
      variantchar_list_                        \
    },                                         \
                                               \
    format_,                                   \
    validate_,                                 \
    get_cmap_info_                             \
  };


  typedef struct  TT_ValidatorRec_
  {
    FT_ValidatorRec  validator;
    FT_UInt          num_glyphs;

  } TT_ValidatorRec, *TT_Validator;


#define TT_VALIDATOR( x )          ( (TT_Validator)( x ) )
#define TT_VALID_GLYPH_COUNT( x )  TT_VALIDATOR( x )->num_glyphs


  FT_CALLBACK_TABLE const TT_CMap_ClassRec  tt_cmap_unicode_class_rec;

  FT_LOCAL( FT_Error )
  tt_face_build_cmaps( TT_Face  face );

  /* used in tt-cmaps service */
  FT_LOCAL( FT_Error )
  tt_get_cmap_info( FT_CharMap    charmap,
                    TT_CMapInfo  *cmap_info );


FT_END_HEADER

#endif /* TTCMAP_H_ */


/* END */
