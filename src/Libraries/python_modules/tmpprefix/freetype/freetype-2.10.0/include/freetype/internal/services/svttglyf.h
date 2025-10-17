/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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
#ifndef SVTTGLYF_H_
#define SVTTGLYF_H_

#include FT_INTERNAL_SERVICE_H
#include FT_TRUETYPE_TABLES_H


FT_BEGIN_HEADER


#define FT_SERVICE_ID_TT_GLYF  "tt-glyf"


  typedef FT_ULong
  (*TT_Glyf_GetLocationFunc)( FT_Face    face,
                              FT_UInt    gindex,
                              FT_ULong  *psize );

  FT_DEFINE_SERVICE( TTGlyf )
  {
    TT_Glyf_GetLocationFunc  get_location;
  };


#define FT_DEFINE_SERVICE_TTGLYFREC( class_, get_location_ )  \
  static const FT_Service_TTGlyfRec  class_ =                 \
  {                                                           \
    get_location_                                             \
  };

  /* */


FT_END_HEADER

#endif /* SVTTGLYF_H_ */


/* END */
