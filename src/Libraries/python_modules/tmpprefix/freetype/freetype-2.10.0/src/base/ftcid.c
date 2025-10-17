/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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
#include <ft2build.h>
#include FT_CID_H
#include FT_INTERNAL_OBJECTS_H
#include FT_SERVICE_CID_H


  /* documentation is in ftcid.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_CID_Registry_Ordering_Supplement( FT_Face       face,
                                           const char*  *registry,
                                           const char*  *ordering,
                                           FT_Int       *supplement)
  {
    FT_Error     error;
    const char*  r = NULL;
    const char*  o = NULL;
    FT_Int       s = 0;


    error = FT_ERR( Invalid_Argument );

    if ( face )
    {
      FT_Service_CID  service;


      FT_FACE_FIND_SERVICE( face, service, CID );

      if ( service && service->get_ros )
        error = service->get_ros( face, &r, &o, &s );
    }

    if ( registry )
      *registry = r;

    if ( ordering )
      *ordering = o;

    if ( supplement )
      *supplement = s;

    return error;
  }


  FT_EXPORT_DEF( FT_Error )
  FT_Get_CID_Is_Internally_CID_Keyed( FT_Face   face,
                                      FT_Bool  *is_cid )
  {
    FT_Error  error = FT_ERR( Invalid_Argument );
    FT_Bool   ic = 0;


    if ( face )
    {
      FT_Service_CID  service;


      FT_FACE_FIND_SERVICE( face, service, CID );

      if ( service && service->get_is_cid )
        error = service->get_is_cid( face, &ic);
    }

    if ( is_cid )
      *is_cid = ic;

    return error;
  }


  FT_EXPORT_DEF( FT_Error )
  FT_Get_CID_From_Glyph_Index( FT_Face   face,
                               FT_UInt   glyph_index,
                               FT_UInt  *cid )
  {
    FT_Error  error = FT_ERR( Invalid_Argument );
    FT_UInt   c = 0;


    if ( face )
    {
      FT_Service_CID  service;


      FT_FACE_FIND_SERVICE( face, service, CID );

      if ( service && service->get_cid_from_glyph_index )
        error = service->get_cid_from_glyph_index( face, glyph_index, &c);
    }

    if ( cid )
      *cid = c;

    return error;
  }


/* END */
