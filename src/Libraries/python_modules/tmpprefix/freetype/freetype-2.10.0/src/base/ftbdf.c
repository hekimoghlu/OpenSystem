/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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
#include FT_INTERNAL_DEBUG_H

#include FT_INTERNAL_OBJECTS_H
#include FT_SERVICE_BDF_H


  /* documentation is in ftbdf.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_BDF_Charset_ID( FT_Face       face,
                         const char*  *acharset_encoding,
                         const char*  *acharset_registry )
  {
    FT_Error     error;
    const char*  encoding = NULL;
    const char*  registry = NULL;

    FT_Service_BDF  service;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    FT_FACE_FIND_SERVICE( face, service, BDF );

    if ( service && service->get_charset_id )
      error = service->get_charset_id( face, &encoding, &registry );
    else
      error = FT_THROW( Invalid_Argument );

    if ( acharset_encoding )
      *acharset_encoding = encoding;

    if ( acharset_registry )
      *acharset_registry = registry;

    return error;
  }


  /* documentation is in ftbdf.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_BDF_Property( FT_Face           face,
                       const char*       prop_name,
                       BDF_PropertyRec  *aproperty )
  {
    FT_Error  error;

    FT_Service_BDF  service;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( !aproperty )
      return FT_THROW( Invalid_Argument );

    aproperty->type = BDF_PROPERTY_TYPE_NONE;

    FT_FACE_FIND_SERVICE( face, service, BDF );

    if ( service && service->get_property )
      error = service->get_property( face, prop_name, aproperty );
    else
      error = FT_THROW( Invalid_Argument );

    return error;
  }


/* END */
