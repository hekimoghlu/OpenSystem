/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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
#ifndef PFRCMAP_H_
#define PFRCMAP_H_

#include <ft2build.h>
#include FT_INTERNAL_OBJECTS_H
#include "pfrtypes.h"


FT_BEGIN_HEADER

  typedef struct  PFR_CMapRec_
  {
    FT_CMapRec  cmap;
    FT_UInt     num_chars;
    PFR_Char    chars;

  } PFR_CMapRec, *PFR_CMap;


  FT_CALLBACK_TABLE const FT_CMap_ClassRec  pfr_cmap_class_rec;

FT_END_HEADER


#endif /* PFRCMAP_H_ */


/* END */
