/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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


#include "gxvmorx.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  gxvmorx


  static void
  gxv_morx_subtable_type0_entry_validate(
    FT_UShort                        state,
    FT_UShort                        flags,
    GXV_XStateTable_GlyphOffsetCPtr  glyphOffset_p,
    FT_Bytes                         table,
    FT_Bytes                         limit,
    GXV_Validator                    gxvalid )
  {
#ifdef GXV_LOAD_UNUSED_VARS
    FT_UShort  markFirst;
    FT_UShort  dontAdvance;
    FT_UShort  markLast;
#endif
    FT_UShort  reserved;
#ifdef GXV_LOAD_UNUSED_VARS
    FT_UShort  verb;
#endif

    FT_UNUSED( state );
    FT_UNUSED( glyphOffset_p );
    FT_UNUSED( table );
    FT_UNUSED( limit );


#ifdef GXV_LOAD_UNUSED_VARS
    markFirst   = (FT_UShort)( ( flags >> 15 ) & 1 );
    dontAdvance = (FT_UShort)( ( flags >> 14 ) & 1 );
    markLast    = (FT_UShort)( ( flags >> 13 ) & 1 );
#endif

    reserved = (FT_UShort)( flags & 0x1FF0 );
#ifdef GXV_LOAD_UNUSED_VARS
    verb     = (FT_UShort)( flags & 0x000F );
#endif

    if ( 0 < reserved )
    {
      GXV_TRACE(( " non-zero bits found in reserved range\n" ));
      FT_INVALID_DATA;
    }
  }


  FT_LOCAL_DEF( void )
  gxv_morx_subtable_type0_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid )
  {
    FT_Bytes  p = table;


    GXV_NAME_ENTER(
      "morx chain subtable type0 (Indic-Script Rearrangement)" );

    GXV_LIMIT_CHECK( GXV_STATETABLE_HEADER_SIZE );

    gxvalid->xstatetable.optdata               = NULL;
    gxvalid->xstatetable.optdata_load_func     = NULL;
    gxvalid->xstatetable.subtable_setup_func   = NULL;
    gxvalid->xstatetable.entry_glyphoffset_fmt = GXV_GLYPHOFFSET_NONE;
    gxvalid->xstatetable.entry_validate_func =
      gxv_morx_subtable_type0_entry_validate;

    gxv_XStateTable_validate( p, limit, gxvalid );

    GXV_EXIT;
  }


/* END */
