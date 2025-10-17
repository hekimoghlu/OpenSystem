/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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


#ifndef SVGXVAL_H_
#define SVGXVAL_H_

#include FT_GX_VALIDATE_H
#include FT_INTERNAL_VALIDATE_H

FT_BEGIN_HEADER


#define FT_SERVICE_ID_GX_VALIDATE           "truetypegx-validate"
#define FT_SERVICE_ID_CLASSICKERN_VALIDATE  "classickern-validate"

  typedef FT_Error
  (*gxv_validate_func)( FT_Face   face,
                        FT_UInt   gx_flags,
                        FT_Bytes  tables[FT_VALIDATE_GX_LENGTH],
                        FT_UInt   table_length );


  typedef FT_Error
  (*ckern_validate_func)( FT_Face   face,
                          FT_UInt   ckern_flags,
                          FT_Bytes  *ckern_table );


  FT_DEFINE_SERVICE( GXvalidate )
  {
    gxv_validate_func  validate;
  };

  FT_DEFINE_SERVICE( CKERNvalidate )
  {
    ckern_validate_func  validate;
  };

  /* */


FT_END_HEADER


#endif /* SVGXVAL_H_ */


/* END */
