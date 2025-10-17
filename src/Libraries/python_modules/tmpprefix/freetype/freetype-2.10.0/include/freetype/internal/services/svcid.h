/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 10, 2025.
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
#ifndef SVCID_H_
#define SVCID_H_

#include FT_INTERNAL_SERVICE_H


FT_BEGIN_HEADER


#define FT_SERVICE_ID_CID  "CID"

  typedef FT_Error
  (*FT_CID_GetRegistryOrderingSupplementFunc)( FT_Face       face,
                                               const char*  *registry,
                                               const char*  *ordering,
                                               FT_Int       *supplement );
  typedef FT_Error
  (*FT_CID_GetIsInternallyCIDKeyedFunc)( FT_Face   face,
                                         FT_Bool  *is_cid );
  typedef FT_Error
  (*FT_CID_GetCIDFromGlyphIndexFunc)( FT_Face   face,
                                      FT_UInt   glyph_index,
                                      FT_UInt  *cid );

  FT_DEFINE_SERVICE( CID )
  {
    FT_CID_GetRegistryOrderingSupplementFunc  get_ros;
    FT_CID_GetIsInternallyCIDKeyedFunc        get_is_cid;
    FT_CID_GetCIDFromGlyphIndexFunc           get_cid_from_glyph_index;
  };


#define FT_DEFINE_SERVICE_CIDREC( class_,                                   \
                                  get_ros_,                                 \
                                  get_is_cid_,                              \
                                  get_cid_from_glyph_index_ )               \
  static const FT_Service_CIDRec class_ =                                   \
  {                                                                         \
    get_ros_, get_is_cid_, get_cid_from_glyph_index_                        \
  };

  /* */


FT_END_HEADER


#endif /* SVCID_H_ */


/* END */
