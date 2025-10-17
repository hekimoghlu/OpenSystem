/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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
#ifndef BDFDRIVR_H_
#define BDFDRIVR_H_

#include <ft2build.h>
#include FT_INTERNAL_DRIVER_H

#include "bdf.h"


FT_BEGIN_HEADER


  typedef struct  BDF_encoding_el_
  {
    FT_ULong   enc;
    FT_UShort  glyph;

  } BDF_encoding_el;


  typedef struct  BDF_FaceRec_
  {
    FT_FaceRec        root;

    char*             charset_encoding;
    char*             charset_registry;

    bdf_font_t*       bdffont;

    BDF_encoding_el*  en_table;

    FT_UInt           default_glyph;

  } BDF_FaceRec, *BDF_Face;


  FT_EXPORT_VAR( const FT_Driver_ClassRec )  bdf_driver_class;


FT_END_HEADER


#endif /* BDFDRIVR_H_ */


/* END */
