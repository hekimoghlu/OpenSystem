/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 31, 2024.
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
/****************************************************************************
 *
 * gxvalid is derived from both gxlayout module and otvalid module.
 * Development of gxlayout is supported by the Information-technology
 * Promotion Agency(IPA), Japan.
 *
 */


#ifndef GXVMORT_H_
#define GXVMORT_H_

#include "gxvalid.h"
#include "gxvcommn.h"

#include FT_SFNT_NAMES_H


  typedef struct  GXV_mort_featureRec_
  {
    FT_UShort  featureType;
    FT_UShort  featureSetting;
    FT_ULong   enableFlags;
    FT_ULong   disableFlags;

  } GXV_mort_featureRec, *GXV_mort_feature;

#define GXV_MORT_FEATURE_OFF  {0, 1, 0x00000000UL, 0x00000000UL}

#define IS_GXV_MORT_FEATURE_OFF( f )              \
          ( (f).featureType    == 0            || \
            (f).featureSetting == 1            || \
            (f).enableFlags    == 0x00000000UL || \
            (f).disableFlags   == 0x00000000UL )


  FT_LOCAL( void )
  gxv_mort_featurearray_validate( FT_Bytes       table,
                                  FT_Bytes       limit,
                                  FT_ULong       nFeatureFlags,
                                  GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_mort_coverage_validate( FT_UShort      coverage,
                              GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_mort_subtable_type0_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_mort_subtable_type1_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_mort_subtable_type2_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_mort_subtable_type4_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_mort_subtable_type5_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );


#endif /* GXVMORT_H_ */


/* END */
