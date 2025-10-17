/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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
#include "aftypes.h"
#include "aflatin.h"
#include "afcjk.h"


#ifdef AF_CONFIG_OPTION_INDIC

#include "afindic.h"
#include "aferrors.h"


#ifdef AF_CONFIG_OPTION_USE_WARPER
#include "afwarp.h"
#endif


  static FT_Error
  af_indic_metrics_init( AF_CJKMetrics  metrics,
                         FT_Face        face )
  {
    /* skip blue zone init in CJK routines */
    FT_CharMap  oldmap = face->charmap;


    metrics->units_per_em = face->units_per_EM;

    if ( FT_Select_Charmap( face, FT_ENCODING_UNICODE ) )
      face->charmap = NULL;
    else
    {
      af_cjk_metrics_init_widths( metrics, face );
#if 0
      /* either need indic specific blue_chars[] or just skip blue zones */
      af_cjk_metrics_init_blues( metrics, face, af_cjk_blue_chars );
#endif
      af_cjk_metrics_check_digits( metrics, face );
    }

    FT_Set_Charmap( face, oldmap );

    return FT_Err_Ok;
  }


  static void
  af_indic_metrics_scale( AF_CJKMetrics  metrics,
                          AF_Scaler      scaler )
  {
    /* use CJK routines */
    af_cjk_metrics_scale( metrics, scaler );
  }


  static FT_Error
  af_indic_hints_init( AF_GlyphHints  hints,
                       AF_CJKMetrics  metrics )
  {
    /* use CJK routines */
    return af_cjk_hints_init( hints, metrics );
  }


  static FT_Error
  af_indic_hints_apply( FT_UInt        glyph_index,
                        AF_GlyphHints  hints,
                        FT_Outline*    outline,
                        AF_CJKMetrics  metrics )
  {
    /* use CJK routines */
    return af_cjk_hints_apply( glyph_index, hints, outline, metrics );
  }


  /* Extract standard_width from writing system/script specific */
  /* metrics class.                                             */

  static void
  af_indic_get_standard_widths( AF_CJKMetrics  metrics,
                                FT_Pos*        stdHW,
                                FT_Pos*        stdVW )
  {
    if ( stdHW )
      *stdHW = metrics->axis[AF_DIMENSION_VERT].standard_width;

    if ( stdVW )
      *stdVW = metrics->axis[AF_DIMENSION_HORZ].standard_width;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                I N D I C   S C R I P T   C L A S S            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  AF_DEFINE_WRITING_SYSTEM_CLASS(
    af_indic_writing_system_class,

    AF_WRITING_SYSTEM_INDIC,

    sizeof ( AF_CJKMetricsRec ),

    (AF_WritingSystem_InitMetricsFunc) af_indic_metrics_init,        /* style_metrics_init    */
    (AF_WritingSystem_ScaleMetricsFunc)af_indic_metrics_scale,       /* style_metrics_scale   */
    (AF_WritingSystem_DoneMetricsFunc) NULL,                         /* style_metrics_done    */
    (AF_WritingSystem_GetStdWidthsFunc)af_indic_get_standard_widths, /* style_metrics_getstdw */

    (AF_WritingSystem_InitHintsFunc)   af_indic_hints_init,          /* style_hints_init      */
    (AF_WritingSystem_ApplyHintsFunc)  af_indic_hints_apply          /* style_hints_apply     */
  )


#else /* !AF_CONFIG_OPTION_INDIC */


  AF_DEFINE_WRITING_SYSTEM_CLASS(
    af_indic_writing_system_class,

    AF_WRITING_SYSTEM_INDIC,

    sizeof ( AF_CJKMetricsRec ),

    (AF_WritingSystem_InitMetricsFunc) NULL, /* style_metrics_init    */
    (AF_WritingSystem_ScaleMetricsFunc)NULL, /* style_metrics_scale   */
    (AF_WritingSystem_DoneMetricsFunc) NULL, /* style_metrics_done    */
    (AF_WritingSystem_GetStdWidthsFunc)NULL, /* style_metrics_getstdw */

    (AF_WritingSystem_InitHintsFunc)   NULL, /* style_hints_init      */
    (AF_WritingSystem_ApplyHintsFunc)  NULL  /* style_hints_apply     */
  )


#endif /* !AF_CONFIG_OPTION_INDIC */


/* END */
