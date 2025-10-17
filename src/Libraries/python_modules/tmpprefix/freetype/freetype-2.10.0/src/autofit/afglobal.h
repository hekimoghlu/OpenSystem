/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
#ifndef AFGLOBAL_H_
#define AFGLOBAL_H_


#include "aftypes.h"
#include "afmodule.h"
#include "afshaper.h"


FT_BEGIN_HEADER


  FT_LOCAL_ARRAY( AF_WritingSystemClass )
  af_writing_system_classes[];


#undef  SCRIPT
#define SCRIPT( s, S, d, h, H, ss )                            \
          AF_DECLARE_SCRIPT_CLASS( af_ ## s ## _script_class )

#include "afscript.h"

  FT_LOCAL_ARRAY( AF_ScriptClass )
  af_script_classes[];


#undef  STYLE
#define STYLE( s, S, d, ws, sc, ss, c )                      \
          AF_DECLARE_STYLE_CLASS( af_ ## s ## _style_class )

#include "afstyles.h"

  FT_LOCAL_ARRAY( AF_StyleClass )
  af_style_classes[];


#ifdef FT_DEBUG_LEVEL_TRACE
  FT_LOCAL_ARRAY( char* )
  af_style_names[];
#endif


  /*
   * Default values and flags for both autofitter globals (found in
   * AF_ModuleRec) and face globals (in AF_FaceGlobalsRec).
   */

  /* index of fallback style in `af_style_classes' */
#ifdef AF_CONFIG_OPTION_CJK
#define AF_STYLE_FALLBACK    AF_STYLE_HANI_DFLT
#else
#define AF_STYLE_FALLBACK    AF_STYLE_NONE_DFLT
#endif
  /* default script for OpenType; ignored if HarfBuzz isn't used */
#define AF_SCRIPT_DEFAULT    AF_SCRIPT_LATN

  /* a bit mask for AF_DIGIT and AF_NONBASE */
#define AF_STYLE_MASK        0x3FFF
  /* an uncovered glyph      */
#define AF_STYLE_UNASSIGNED  AF_STYLE_MASK

  /* if this flag is set, we have an ASCII digit   */
#define AF_DIGIT             0x8000U
  /* if this flag is set, we have a non-base character */
#define AF_NONBASE           0x4000U

  /* `increase-x-height' property */
#define AF_PROP_INCREASE_X_HEIGHT_MIN  6
#define AF_PROP_INCREASE_X_HEIGHT_MAX  0


  /************************************************************************/
  /************************************************************************/
  /*****                                                              *****/
  /*****                  F A C E   G L O B A L S                     *****/
  /*****                                                              *****/
  /************************************************************************/
  /************************************************************************/


  /*
   * Note that glyph_styles[] maps each glyph to an index into the
   * `af_style_classes' array.
   *
   */
  typedef struct  AF_FaceGlobalsRec_
  {
    FT_Face          face;
    FT_Long          glyph_count;    /* same as face->num_glyphs */
    FT_UShort*       glyph_styles;

#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ
    hb_font_t*       hb_font;
    hb_buffer_t*     hb_buf;           /* for feature comparison */
#endif

    /* per-face auto-hinter properties */
    FT_UInt          increase_x_height;

    AF_StyleMetrics  metrics[AF_STYLE_MAX];

    /* Compute darkening amount once per size.  Use this to check whether */
    /* darken_{x,y} needs to be recomputed.                               */
    FT_UShort        stem_darkening_for_ppem;
    /* Copy from e.g. AF_LatinMetrics.axis[AF_DIMENSION_HORZ] */
    /* to compute the darkening amount.                       */
    FT_Pos           standard_vertical_width;
    /* Copy from e.g. AF_LatinMetrics.axis[AF_DIMENSION_VERT] */
    /* to compute the darkening amount.                       */
    FT_Pos           standard_horizontal_width;
    /* The actual amount to darken a glyph along the X axis. */
    FT_Pos           darken_x;
    /* The actual amount to darken a glyph along the Y axis. */
    FT_Pos           darken_y;
    /* Amount to scale down by to keep emboldened points */
    /* on the Y-axis in pre-computed blue zones.         */
    FT_Fixed         scale_down_factor;
    AF_Module        module;         /* to access global properties */

  } AF_FaceGlobalsRec;


  /*
   * model the global hints data for a given face, decomposed into
   * style-specific items
   */

  FT_LOCAL( FT_Error )
  af_face_globals_new( FT_Face          face,
                       AF_FaceGlobals  *aglobals,
                       AF_Module        module );

  FT_LOCAL( FT_Error )
  af_face_globals_get_metrics( AF_FaceGlobals    globals,
                               FT_UInt           gindex,
                               FT_UInt           options,
                               AF_StyleMetrics  *ametrics );

  FT_LOCAL( void )
  af_face_globals_free( AF_FaceGlobals  globals );

  FT_LOCAL_DEF( FT_Bool )
  af_face_globals_is_digit( AF_FaceGlobals  globals,
                            FT_UInt         gindex );

  /* */


FT_END_HEADER

#endif /* AFGLOBAL_H_ */


/* END */
