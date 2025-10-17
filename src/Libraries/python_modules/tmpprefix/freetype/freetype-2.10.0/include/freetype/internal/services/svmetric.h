/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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
#ifndef SVMETRIC_H_
#define SVMETRIC_H_

#include FT_INTERNAL_SERVICE_H


FT_BEGIN_HEADER


  /*
   * A service to manage the `HVAR, `MVAR', and `VVAR' OpenType tables.
   *
   */

#define FT_SERVICE_ID_METRICS_VARIATIONS  "metrics-variations"


  /* HVAR */

  typedef FT_Error
  (*FT_HAdvance_Adjust_Func)( FT_Face  face,
                              FT_UInt  gindex,
                              FT_Int  *avalue );

  typedef FT_Error
  (*FT_LSB_Adjust_Func)( FT_Face  face,
                         FT_UInt  gindex,
                         FT_Int  *avalue );

  typedef FT_Error
  (*FT_RSB_Adjust_Func)( FT_Face  face,
                         FT_UInt  gindex,
                         FT_Int  *avalue );

  /* VVAR */

  typedef FT_Error
  (*FT_VAdvance_Adjust_Func)( FT_Face  face,
                              FT_UInt  gindex,
                              FT_Int  *avalue );

  typedef FT_Error
  (*FT_TSB_Adjust_Func)( FT_Face  face,
                         FT_UInt  gindex,
                         FT_Int  *avalue );

  typedef FT_Error
  (*FT_BSB_Adjust_Func)( FT_Face  face,
                         FT_UInt  gindex,
                         FT_Int  *avalue );

  typedef FT_Error
  (*FT_VOrg_Adjust_Func)( FT_Face  face,
                          FT_UInt  gindex,
                          FT_Int  *avalue );

  /* MVAR */

  typedef void
  (*FT_Metrics_Adjust_Func)( FT_Face  face );


  FT_DEFINE_SERVICE( MetricsVariations )
  {
    FT_HAdvance_Adjust_Func  hadvance_adjust;
    FT_LSB_Adjust_Func       lsb_adjust;
    FT_RSB_Adjust_Func       rsb_adjust;

    FT_VAdvance_Adjust_Func  vadvance_adjust;
    FT_TSB_Adjust_Func       tsb_adjust;
    FT_BSB_Adjust_Func       bsb_adjust;
    FT_VOrg_Adjust_Func      vorg_adjust;

    FT_Metrics_Adjust_Func   metrics_adjust;
  };


#define FT_DEFINE_SERVICE_METRICSVARIATIONSREC( class_,            \
                                                hadvance_adjust_,  \
                                                lsb_adjust_,       \
                                                rsb_adjust_,       \
                                                vadvance_adjust_,  \
                                                tsb_adjust_,       \
                                                bsb_adjust_,       \
                                                vorg_adjust_,      \
                                                metrics_adjust_  ) \
  static const FT_Service_MetricsVariationsRec  class_ =           \
  {                                                                \
    hadvance_adjust_,                                              \
    lsb_adjust_,                                                   \
    rsb_adjust_,                                                   \
    vadvance_adjust_,                                              \
    tsb_adjust_,                                                   \
    bsb_adjust_,                                                   \
    vorg_adjust_,                                                  \
    metrics_adjust_                                                \
  };

  /* */


FT_END_HEADER

#endif /* SVMETRIC_H_ */


/* END */
