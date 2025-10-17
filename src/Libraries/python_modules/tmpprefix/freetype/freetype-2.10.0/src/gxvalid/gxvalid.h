/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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


#ifndef GXVALID_H_
#define GXVALID_H_

#include <ft2build.h>
#include FT_FREETYPE_H

#include "gxverror.h"          /* must come before FT_INTERNAL_VALIDATE_H */

#include FT_INTERNAL_VALIDATE_H
#include FT_INTERNAL_STREAM_H


FT_BEGIN_HEADER


  FT_LOCAL( void )
  gxv_feat_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );


  FT_LOCAL( void )
  gxv_bsln_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );


  FT_LOCAL( void )
  gxv_trak_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_just_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_mort_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_morx_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_kern_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_kern_validate_classic( FT_Bytes      table,
                             FT_Face       face,
                             FT_Int        dialect_flags,
                             FT_Validator  valid );

  FT_LOCAL( void )
  gxv_opbd_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_prop_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_lcar_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );


FT_END_HEADER


#endif /* GXVALID_H_ */


/* END */
