/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 23, 2023.
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
/**************************************************************************
   *
   * This component has a _single_ role: to compute exact outline bounding
   * boxes.
   *
   * It is separated from the rest of the engine for various technical
   * reasons.  It may well be integrated in 'ftoutln' later.
   *
   */


#ifndef FTBBOX_H_
#define FTBBOX_H_


#include <ft2build.h>
#include FT_FREETYPE_H

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @section:
   *   outline_processing
   *
   */


  /**************************************************************************
   *
   * @function:
   *   FT_Outline_Get_BBox
   *
   * @description:
   *   Compute the exact bounding box of an outline.  This is slower than
   *   computing the control box.  However, it uses an advanced algorithm
   *   that returns _very_ quickly when the two boxes coincide.  Otherwise,
   *   the outline Bezier arcs are traversed to extract their extrema.
   *
   * @input:
   *   outline ::
   *     A pointer to the source outline.
   *
   * @output:
   *   abbox ::
   *     The outline's exact bounding box.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   If the font is tricky and the glyph has been loaded with
   *   @FT_LOAD_NO_SCALE, the resulting BBox is meaningless.  To get
   *   reasonable values for the BBox it is necessary to load the glyph at a
   *   large ppem value (so that the hinting instructions can properly shift
   *   and scale the subglyphs), then extracting the BBox, which can be
   *   eventually converted back to font units.
   */
  FT_EXPORT( FT_Error )
  FT_Outline_Get_BBox( FT_Outline*  outline,
                       FT_BBox     *abbox );

  /* */


FT_END_HEADER

#endif /* FTBBOX_H_ */


/* END */


/* Local Variables: */
/* coding: utf-8    */
/* End:             */
