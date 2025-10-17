/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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


#include "gxvmort.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  gxvmort


  static void
  gxv_mort_subtable_type4_lookupval_validate( FT_UShort            glyph,
                                              GXV_LookupValueCPtr  value_p,
                                              GXV_Validator        gxvalid )
  {
    FT_UNUSED( glyph );

    gxv_glyphid_validate( value_p->u, gxvalid );
  }

  /*
    +===============+ --------+
    | lookup header |         |
    +===============+         |
    | BinSrchHeader |         |
    +===============+         |
    | lastGlyph[0]  |         |
    +---------------+         |
    | firstGlyph[0] |         |    head of lookup table
    +---------------+         |             +
    | offset[0]     |    ->   |          offset            [byte]
    +===============+         |             +
    | lastGlyph[1]  |         | (glyphID - firstGlyph) * 2 [byte]
    +---------------+         |
    | firstGlyph[1] |         |
    +---------------+         |
    | offset[1]     |         |
    +===============+         |
                              |
     ....                     |
                              |
    16bit value array         |
    +===============+         |
    |     value     | <-------+
     ....
  */

  static GXV_LookupValueDesc
  gxv_mort_subtable_type4_lookupfmt4_transit(
    FT_UShort            relative_gindex,
    GXV_LookupValueCPtr  base_value_p,
    FT_Bytes             lookuptbl_limit,
    GXV_Validator        gxvalid )
  {
    FT_Bytes             p;
    FT_Bytes             limit;
    FT_UShort            offset;
    GXV_LookupValueDesc  value;

    /* XXX: check range? */
    offset = (FT_UShort)( base_value_p->u +
                          relative_gindex * sizeof ( FT_UShort ) );

    p     = gxvalid->lookuptbl_head + offset;
    limit = lookuptbl_limit;

    GXV_LIMIT_CHECK( 2 );
    value.u = FT_NEXT_USHORT( p );

    return value;
  }


  FT_LOCAL_DEF( void )
  gxv_mort_subtable_type4_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid )
  {
    FT_Bytes  p = table;


    GXV_NAME_ENTER( "mort chain subtable type4 "
                    "(Non-Contextual Glyph Substitution)" );

    gxvalid->lookupval_sign   = GXV_LOOKUPVALUE_UNSIGNED;
    gxvalid->lookupval_func   = gxv_mort_subtable_type4_lookupval_validate;
    gxvalid->lookupfmt4_trans = gxv_mort_subtable_type4_lookupfmt4_transit;

    gxv_LookupTable_validate( p, limit, gxvalid );

    GXV_EXIT;
  }


/* END */
