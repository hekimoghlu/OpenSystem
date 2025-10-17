/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef SVPFR_H_
#define SVPFR_H_

#include FT_PFR_H
#include FT_INTERNAL_SERVICE_H


FT_BEGIN_HEADER


#define FT_SERVICE_ID_PFR_METRICS  "pfr-metrics"


  typedef FT_Error
  (*FT_PFR_GetMetricsFunc)( FT_Face    face,
                            FT_UInt   *aoutline,
                            FT_UInt   *ametrics,
                            FT_Fixed  *ax_scale,
                            FT_Fixed  *ay_scale );

  typedef FT_Error
  (*FT_PFR_GetKerningFunc)( FT_Face     face,
                            FT_UInt     left,
                            FT_UInt     right,
                            FT_Vector  *avector );

  typedef FT_Error
  (*FT_PFR_GetAdvanceFunc)( FT_Face   face,
                            FT_UInt   gindex,
                            FT_Pos   *aadvance );


  FT_DEFINE_SERVICE( PfrMetrics )
  {
    FT_PFR_GetMetricsFunc  get_metrics;
    FT_PFR_GetKerningFunc  get_kerning;
    FT_PFR_GetAdvanceFunc  get_advance;

  };

 /* */

FT_END_HEADER

#endif /* SVPFR_H_ */


/* END */
