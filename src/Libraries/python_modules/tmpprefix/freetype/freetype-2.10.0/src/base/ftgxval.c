/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
/****************************************************************************
 *
 * gxvalid is derived from both gxlayout module and otvalid module.
 * Development of gxlayout is supported by the Information-technology
 * Promotion Agency(IPA), Japan.
 *
 */


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H

#include FT_INTERNAL_OBJECTS_H
#include FT_SERVICE_GX_VALIDATE_H


  /* documentation is in ftgxval.h */

  FT_EXPORT_DEF( FT_Error )
  FT_TrueTypeGX_Validate( FT_Face   face,
                          FT_UInt   validation_flags,
                          FT_Bytes  tables[FT_VALIDATE_GX_LENGTH],
                          FT_UInt   table_length )
  {
    FT_Service_GXvalidate  service;
    FT_Error               error;


    if ( !face )
    {
      error = FT_THROW( Invalid_Face_Handle );
      goto Exit;
    }

    if ( !tables )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    FT_FACE_FIND_GLOBAL_SERVICE( face, service, GX_VALIDATE );

    if ( service )
      error = service->validate( face,
                                 validation_flags,
                                 tables,
                                 table_length );
    else
      error = FT_THROW( Unimplemented_Feature );

  Exit:
    return error;
  }


  FT_EXPORT_DEF( void )
  FT_TrueTypeGX_Free( FT_Face   face,
                      FT_Bytes  table )
  {
    FT_Memory  memory;


    if ( !face )
      return;

    memory = FT_FACE_MEMORY( face );

    FT_FREE( table );
  }


  FT_EXPORT_DEF( FT_Error )
  FT_ClassicKern_Validate( FT_Face    face,
                           FT_UInt    validation_flags,
                           FT_Bytes  *ckern_table )
  {
    FT_Service_CKERNvalidate  service;
    FT_Error                  error;


    if ( !face )
    {
      error = FT_THROW( Invalid_Face_Handle );
      goto Exit;
    }

    if ( !ckern_table )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    FT_FACE_FIND_GLOBAL_SERVICE( face, service, CLASSICKERN_VALIDATE );

    if ( service )
      error = service->validate( face,
                                 validation_flags,
                                 ckern_table );
    else
      error = FT_THROW( Unimplemented_Feature );

  Exit:
    return error;
  }


  FT_EXPORT_DEF( void )
  FT_ClassicKern_Free( FT_Face   face,
                       FT_Bytes  table )
  {
    FT_Memory  memory;


    if ( !face )
      return;

    memory = FT_FACE_MEMORY( face );


    FT_FREE( table );
  }


/* END */
