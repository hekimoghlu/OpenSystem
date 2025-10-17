/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
/* Development of this service is support of
   Information-technology Promotion Agency, Japan. */

#ifndef SVTTCMAP_H_
#define SVTTCMAP_H_

#include FT_INTERNAL_SERVICE_H
#include FT_TRUETYPE_TABLES_H


FT_BEGIN_HEADER


#define FT_SERVICE_ID_TT_CMAP  "tt-cmaps"


  /**************************************************************************
   *
   * @struct:
   *   TT_CMapInfo
   *
   * @description:
   *   A structure used to store TrueType/sfnt specific cmap information
   *   which is not covered by the generic @FT_CharMap structure.  This
   *   structure can be accessed with the @FT_Get_TT_CMap_Info function.
   *
   * @fields:
   *   language ::
   *     The language ID used in Mac fonts.  Definitions of values are in
   *     `ttnameid.h`.
   *
   *   format ::
   *     The cmap format.  OpenType 1.6 defines the formats 0 (byte encoding
   *     table), 2~(high-byte mapping through table), 4~(segment mapping to
   *     delta values), 6~(trimmed table mapping), 8~(mixed 16-bit and 32-bit
   *     coverage), 10~(trimmed array), 12~(segmented coverage), 13~(last
   *     resort font), and 14 (Unicode Variation Sequences).
   */
  typedef struct  TT_CMapInfo_
  {
    FT_ULong  language;
    FT_Long   format;

  } TT_CMapInfo;


  typedef FT_Error
  (*TT_CMap_Info_GetFunc)( FT_CharMap    charmap,
                           TT_CMapInfo  *cmap_info );


  FT_DEFINE_SERVICE( TTCMaps )
  {
    TT_CMap_Info_GetFunc  get_cmap_info;
  };


#define FT_DEFINE_SERVICE_TTCMAPSREC( class_, get_cmap_info_ )  \
  static const FT_Service_TTCMapsRec  class_ =                  \
  {                                                             \
    get_cmap_info_                                              \
  };

  /* */


FT_END_HEADER

#endif /* SVTTCMAP_H_ */


/* END */
