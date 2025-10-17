/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#ifndef FTPFR_H_
#define FTPFR_H_

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
   *   pfr_fonts
   *
   * @title:
   *   PFR Fonts
   *
   * @abstract:
   *   PFR/TrueDoc-specific API.
   *
   * @description:
   *   This section contains the declaration of PFR-specific functions.
   *
   */


  /**************************************************************************
   *
   * @function:
   *    FT_Get_PFR_Metrics
   *
   * @description:
   *    Return the outline and metrics resolutions of a given PFR face.
   *
   * @input:
   *    face ::
   *      Handle to the input face.  It can be a non-PFR face.
   *
   * @output:
   *    aoutline_resolution ::
   *      Outline resolution.  This is equivalent to `face->units_per_EM` for
   *      non-PFR fonts.  Optional (parameter can be `NULL`).
   *
   *    ametrics_resolution ::
   *      Metrics resolution.  This is equivalent to `outline_resolution` for
   *      non-PFR fonts.  Optional (parameter can be `NULL`).
   *
   *    ametrics_x_scale ::
   *      A 16.16 fixed-point number used to scale distance expressed in
   *      metrics units to device subpixels.  This is equivalent to
   *      `face->size->x_scale`, but for metrics only.  Optional (parameter
   *      can be `NULL`).
   *
   *    ametrics_y_scale ::
   *      Same as `ametrics_x_scale` but for the vertical direction.
   *      optional (parameter can be `NULL`).
   *
   * @return:
   *    FreeType error code.  0~means success.
   *
   * @note:
   *   If the input face is not a PFR, this function will return an error.
   *   However, in all cases, it will return valid values.
   */
  FT_EXPORT( FT_Error )
  FT_Get_PFR_Metrics( FT_Face    face,
                      FT_UInt   *aoutline_resolution,
                      FT_UInt   *ametrics_resolution,
                      FT_Fixed  *ametrics_x_scale,
                      FT_Fixed  *ametrics_y_scale );


  /**************************************************************************
   *
   * @function:
   *    FT_Get_PFR_Kerning
   *
   * @description:
   *    Return the kerning pair corresponding to two glyphs in a PFR face.
   *    The distance is expressed in metrics units, unlike the result of
   *    @FT_Get_Kerning.
   *
   * @input:
   *    face ::
   *      A handle to the input face.
   *
   *    left ::
   *      Index of the left glyph.
   *
   *    right ::
   *      Index of the right glyph.
   *
   * @output:
   *    avector ::
   *      A kerning vector.
   *
   * @return:
   *    FreeType error code.  0~means success.
   *
   * @note:
   *    This function always return distances in original PFR metrics units.
   *    This is unlike @FT_Get_Kerning with the @FT_KERNING_UNSCALED mode,
   *    which always returns distances converted to outline units.
   *
   *    You can use the value of the `x_scale` and `y_scale` parameters
   *    returned by @FT_Get_PFR_Metrics to scale these to device subpixels.
   */
  FT_EXPORT( FT_Error )
  FT_Get_PFR_Kerning( FT_Face     face,
                      FT_UInt     left,
                      FT_UInt     right,
                      FT_Vector  *avector );


  /**************************************************************************
   *
   * @function:
   *    FT_Get_PFR_Advance
   *
   * @description:
   *    Return a given glyph advance, expressed in original metrics units,
   *    from a PFR font.
   *
   * @input:
   *    face ::
   *      A handle to the input face.
   *
   *    gindex ::
   *      The glyph index.
   *
   * @output:
   *    aadvance ::
   *      The glyph advance in metrics units.
   *
   * @return:
   *    FreeType error code.  0~means success.
   *
   * @note:
   *    You can use the `x_scale` or `y_scale` results of @FT_Get_PFR_Metrics
   *    to convert the advance to device subpixels (i.e., 1/64th of pixels).
   */
  FT_EXPORT( FT_Error )
  FT_Get_PFR_Advance( FT_Face   face,
                      FT_UInt   gindex,
                      FT_Pos   *aadvance );

  /* */


FT_END_HEADER

#endif /* FTPFR_H_ */


/* END */
