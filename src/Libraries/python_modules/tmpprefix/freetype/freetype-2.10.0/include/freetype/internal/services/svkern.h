/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
#ifndef SVKERN_H_
#define SVKERN_H_

#include FT_INTERNAL_SERVICE_H
#include FT_TRUETYPE_TABLES_H


FT_BEGIN_HEADER

#define FT_SERVICE_ID_KERNING  "kerning"


  typedef FT_Error
  (*FT_Kerning_TrackGetFunc)( FT_Face    face,
                              FT_Fixed   point_size,
                              FT_Int     degree,
                              FT_Fixed*  akerning );

  FT_DEFINE_SERVICE( Kerning )
  {
    FT_Kerning_TrackGetFunc  get_track;
  };

  /* */


FT_END_HEADER


#endif /* SVKERN_H_ */


/* END */
