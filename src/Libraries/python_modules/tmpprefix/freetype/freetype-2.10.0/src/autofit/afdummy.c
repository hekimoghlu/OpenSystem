/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 22, 2025.
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
#include "afdummy.h"
#include "afhints.h"
#include "aferrors.h"


  static FT_Error
  af_dummy_hints_init( AF_GlyphHints    hints,
                       AF_StyleMetrics  metrics )
  {
    af_glyph_hints_rescale( hints, metrics );

    hints->x_scale = metrics->scaler.x_scale;
    hints->y_scale = metrics->scaler.y_scale;
    hints->x_delta = metrics->scaler.x_delta;
    hints->y_delta = metrics->scaler.y_delta;

    return FT_Err_Ok;
  }


  static FT_Error
  af_dummy_hints_apply( FT_UInt          glyph_index,
                        AF_GlyphHints    hints,
                        FT_Outline*      outline,
                        AF_StyleMetrics  metrics )
  {
    FT_Error  error;

    FT_UNUSED( glyph_index );
    FT_UNUSED( metrics );


    error = af_glyph_hints_reload( hints, outline );
    if ( !error )
      af_glyph_hints_save( hints, outline );

    return error;
  }


  AF_DEFINE_WRITING_SYSTEM_CLASS(
    af_dummy_writing_system_class,

    AF_WRITING_SYSTEM_DUMMY,

    sizeof ( AF_StyleMetricsRec ),

    (AF_WritingSystem_InitMetricsFunc) NULL,                /* style_metrics_init    */
    (AF_WritingSystem_ScaleMetricsFunc)NULL,                /* style_metrics_scale   */
    (AF_WritingSystem_DoneMetricsFunc) NULL,                /* style_metrics_done    */
    (AF_WritingSystem_GetStdWidthsFunc)NULL,                /* style_metrics_getstdw */

    (AF_WritingSystem_InitHintsFunc)   af_dummy_hints_init, /* style_hints_init      */
    (AF_WritingSystem_ApplyHintsFunc)  af_dummy_hints_apply /* style_hints_apply     */
  )


/* END */
