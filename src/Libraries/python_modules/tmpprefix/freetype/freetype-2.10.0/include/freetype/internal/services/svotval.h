/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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
#ifndef SVOTVAL_H_
#define SVOTVAL_H_

#include FT_OPENTYPE_VALIDATE_H
#include FT_INTERNAL_VALIDATE_H

FT_BEGIN_HEADER


#define FT_SERVICE_ID_OPENTYPE_VALIDATE  "opentype-validate"


  typedef FT_Error
  (*otv_validate_func)( FT_Face volatile  face,
                        FT_UInt           ot_flags,
                        FT_Bytes         *base,
                        FT_Bytes         *gdef,
                        FT_Bytes         *gpos,
                        FT_Bytes         *gsub,
                        FT_Bytes         *jstf );


  FT_DEFINE_SERVICE( OTvalidate )
  {
    otv_validate_func  validate;
  };

  /* */


FT_END_HEADER


#endif /* SVOTVAL_H_ */


/* END */
