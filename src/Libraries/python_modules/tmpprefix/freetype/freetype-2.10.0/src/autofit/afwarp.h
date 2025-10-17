/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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
#ifndef AFWARP_H_
#define AFWARP_H_

#include "afhints.h"

FT_BEGIN_HEADER

#define AF_WARPER_SCALE

#define AF_WARPER_FLOOR( x )  ( (x) & ~FT_TYPEOF( x )63 )
#define AF_WARPER_CEIL( x )   AF_WARPER_FLOOR( (x) + 63 )


  typedef FT_Int32  AF_WarpScore;

  typedef struct  AF_WarperRec_
  {
    FT_Pos        x1, x2;
    FT_Pos        t1, t2;
    FT_Pos        x1min, x1max;
    FT_Pos        x2min, x2max;
    FT_Pos        w0, wmin, wmax;

    FT_Fixed      best_scale;
    FT_Pos        best_delta;
    AF_WarpScore  best_score;
    AF_WarpScore  best_distort;

  } AF_WarperRec, *AF_Warper;


#ifdef AF_CONFIG_OPTION_USE_WARPER
  FT_LOCAL( void )
  af_warper_compute( AF_Warper      warper,
                     AF_GlyphHints  hints,
                     AF_Dimension   dim,
                     FT_Fixed      *a_scale,
                     FT_Pos        *a_delta );
#endif


FT_END_HEADER


#endif /* AFWARP_H_ */


/* END */
