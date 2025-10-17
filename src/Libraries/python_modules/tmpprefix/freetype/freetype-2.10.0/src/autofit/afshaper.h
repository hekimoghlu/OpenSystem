/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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
#ifndef AFSHAPER_H_
#define AFSHAPER_H_


#include <ft2build.h>
#include FT_FREETYPE_H


#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ

#include <hb.h>
#include <hb-ot.h>
#include <hb-ft.h>

#endif


FT_BEGIN_HEADER

  FT_Error
  af_shaper_get_coverage( AF_FaceGlobals  globals,
                          AF_StyleClass   style_class,
                          FT_UShort*      gstyles,
                          FT_Bool         default_script );


  void*
  af_shaper_buf_create( FT_Face  face );

  void
  af_shaper_buf_destroy( FT_Face  face,
                         void*    buf );

  const char*
  af_shaper_get_cluster( const char*      p,
                         AF_StyleMetrics  metrics,
                         void*            buf_,
                         unsigned int*    count );

  FT_ULong
  af_shaper_get_elem( AF_StyleMetrics  metrics,
                      void*            buf_,
                      unsigned int     idx,
                      FT_Long*         x_advance,
                      FT_Long*         y_offset );

 /* */

FT_END_HEADER

#endif /* AFSHAPER_H_ */


/* END */
