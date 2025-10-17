/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
#ifndef TTBDF_H_
#define TTBDF_H_


#include <ft2build.h>
#include "ttload.h"
#include FT_BDF_H


FT_BEGIN_HEADER


#ifdef TT_CONFIG_OPTION_BDF

  FT_LOCAL( void )
  tt_face_free_bdf_props( TT_Face  face );


  FT_LOCAL( FT_Error )
  tt_face_find_bdf_prop( TT_Face           face,
                         const char*       property_name,
                         BDF_PropertyRec  *aprop );

#endif /* TT_CONFIG_OPTION_BDF */


FT_END_HEADER

#endif /* TTBDF_H_ */


/* END */
