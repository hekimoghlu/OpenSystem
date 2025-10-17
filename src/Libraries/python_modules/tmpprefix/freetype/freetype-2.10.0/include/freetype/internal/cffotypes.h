/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#ifndef CFFOTYPES_H_
#define CFFOTYPES_H_

#include <ft2build.h>
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_CFF_TYPES_H
#include FT_INTERNAL_TRUETYPE_TYPES_H
#include FT_SERVICE_POSTSCRIPT_CMAPS_H
#include FT_INTERNAL_POSTSCRIPT_HINTS_H


FT_BEGIN_HEADER


  typedef TT_Face  CFF_Face;


  /**************************************************************************
   *
   * @type:
   *   CFF_Size
   *
   * @description:
   *   A handle to an OpenType size object.
   */
  typedef struct  CFF_SizeRec_
  {
    FT_SizeRec  root;
    FT_ULong    strike_index;    /* 0xFFFFFFFF to indicate invalid */

  } CFF_SizeRec, *CFF_Size;


  /**************************************************************************
   *
   * @type:
   *   CFF_GlyphSlot
   *
   * @description:
   *   A handle to an OpenType glyph slot object.
   */
  typedef struct  CFF_GlyphSlotRec_
  {
    FT_GlyphSlotRec  root;

    FT_Bool  hint;
    FT_Bool  scaled;

    FT_Fixed  x_scale;
    FT_Fixed  y_scale;

  } CFF_GlyphSlotRec, *CFF_GlyphSlot;


  /**************************************************************************
   *
   * @type:
   *   CFF_Internal
   *
   * @description:
   *   The interface to the 'internal' field of `FT_Size`.
   */
  typedef struct  CFF_InternalRec_
  {
    PSH_Globals  topfont;
    PSH_Globals  subfonts[CFF_MAX_CID_FONTS];

  } CFF_InternalRec, *CFF_Internal;


  /**************************************************************************
   *
   * Subglyph transformation record.
   */
  typedef struct  CFF_Transform_
  {
    FT_Fixed    xx, xy;     /* transformation matrix coefficients */
    FT_Fixed    yx, yy;
    FT_F26Dot6  ox, oy;     /* offsets                            */

  } CFF_Transform;


FT_END_HEADER


#endif /* CFFOTYPES_H_ */


/* END */
