/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#ifndef OTVALID_H_
#define OTVALID_H_


#include <ft2build.h>
#include FT_FREETYPE_H

#include "otverror.h"           /* must come before FT_INTERNAL_VALIDATE_H */

#include FT_INTERNAL_VALIDATE_H
#include FT_INTERNAL_STREAM_H


FT_BEGIN_HEADER


  FT_LOCAL( void )
  otv_BASE_validate( FT_Bytes      table,
                     FT_Validator  valid );

  /* GSUB and GPOS tables should already be validated; */
  /* if missing, set corresponding argument to 0       */
  FT_LOCAL( void )
  otv_GDEF_validate( FT_Bytes      table,
                     FT_Bytes      gsub,
                     FT_Bytes      gpos,
                     FT_UInt       glyph_count,
                     FT_Validator  valid );

  FT_LOCAL( void )
  otv_GPOS_validate( FT_Bytes      table,
                     FT_UInt       glyph_count,
                     FT_Validator  valid );

  FT_LOCAL( void )
  otv_GSUB_validate( FT_Bytes      table,
                     FT_UInt       glyph_count,
                     FT_Validator  valid );

  /* GSUB and GPOS tables should already be validated; */
  /* if missing, set corresponding argument to 0       */
  FT_LOCAL( void )
  otv_JSTF_validate( FT_Bytes      table,
                     FT_Bytes      gsub,
                     FT_Bytes      gpos,
                     FT_UInt       glyph_count,
                     FT_Validator  valid );

  FT_LOCAL( void )
  otv_MATH_validate( FT_Bytes      table,
                     FT_UInt       glyph_count,
                     FT_Validator  ftvalid );


FT_END_HEADER

#endif /* OTVALID_H_ */


/* END */
