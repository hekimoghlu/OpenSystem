/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
#ifndef T1CMAP_H_
#define T1CMAP_H_

#include <ft2build.h>
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_TYPE1_TYPES_H

FT_BEGIN_HEADER


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****          TYPE1 STANDARD (AND EXPERT) ENCODING CMAPS           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* standard (and expert) encoding cmaps */
  typedef struct T1_CMapStdRec_*  T1_CMapStd;

  typedef struct  T1_CMapStdRec_
  {
    FT_CMapRec                cmap;

    const FT_UShort*          code_to_sid;
    PS_Adobe_Std_StringsFunc  sid_to_string;

    FT_UInt                   num_glyphs;
    const char* const*        glyph_names;

  } T1_CMapStdRec;


  FT_CALLBACK_TABLE const FT_CMap_ClassRec
  t1_cmap_standard_class_rec;

  FT_CALLBACK_TABLE const FT_CMap_ClassRec
  t1_cmap_expert_class_rec;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                  TYPE1 CUSTOM ENCODING CMAP                   *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  typedef struct T1_CMapCustomRec_*  T1_CMapCustom;

  typedef struct  T1_CMapCustomRec_
  {
    FT_CMapRec  cmap;
    FT_UInt     first;
    FT_UInt     count;
    FT_UShort*  indices;

  } T1_CMapCustomRec;


  FT_CALLBACK_TABLE const FT_CMap_ClassRec
  t1_cmap_custom_class_rec;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****             TYPE1 SYNTHETIC UNICODE ENCODING CMAP             *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* unicode (synthetic) cmaps */

  FT_CALLBACK_TABLE const FT_CMap_ClassRec
  t1_cmap_unicode_class_rec;

 /* */


FT_END_HEADER

#endif /* T1CMAP_H_ */


/* END */
