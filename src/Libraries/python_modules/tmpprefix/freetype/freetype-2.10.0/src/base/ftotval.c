/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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
#include FT_SERVICE_OPENTYPE_VALIDATE_H
#include FT_OPENTYPE_VALIDATE_H


  /* documentation is in ftotval.h */

  FT_EXPORT_DEF( FT_Error )
  FT_OpenType_Validate( FT_Face    face,
                        FT_UInt    validation_flags,
                        FT_Bytes  *BASE_table,
                        FT_Bytes  *GDEF_table,
                        FT_Bytes  *GPOS_table,
                        FT_Bytes  *GSUB_table,
                        FT_Bytes  *JSTF_table )
  {
    FT_Service_OTvalidate  service;
    FT_Error               error;


    if ( !face )
    {
      error = FT_THROW( Invalid_Face_Handle );
      goto Exit;
    }

    if ( !( BASE_table &&
            GDEF_table &&
            GPOS_table &&
            GSUB_table &&
            JSTF_table ) )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    FT_FACE_FIND_GLOBAL_SERVICE( face, service, OPENTYPE_VALIDATE );

    if ( service )
      error = service->validate( face,
                                 validation_flags,
                                 BASE_table,
                                 GDEF_table,
                                 GPOS_table,
                                 GSUB_table,
                                 JSTF_table );
    else
      error = FT_THROW( Unimplemented_Feature );

  Exit:
    return error;
  }


  FT_EXPORT_DEF( void )
  FT_OpenType_Free( FT_Face   face,
                    FT_Bytes  table )
  {
    FT_Memory  memory;


    if ( !face )
      return;

    memory = FT_FACE_MEMORY( face );

    FT_FREE( table );
  }


/* END */
