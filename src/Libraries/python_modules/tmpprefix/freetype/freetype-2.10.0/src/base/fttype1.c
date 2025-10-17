/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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
#include FT_INTERNAL_SERVICE_H
#include FT_SERVICE_POSTSCRIPT_INFO_H


  /* documentation is in t1tables.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_PS_Font_Info( FT_Face          face,
                       PS_FontInfoRec*  afont_info )
  {
    FT_Error           error;
    FT_Service_PsInfo  service;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( !afont_info )
      return FT_THROW( Invalid_Argument );

    FT_FACE_FIND_SERVICE( face, service, POSTSCRIPT_INFO );

    if ( service && service->ps_get_font_info )
      error = service->ps_get_font_info( face, afont_info );
    else
      error = FT_THROW( Invalid_Argument );

    return error;
  }


  /* documentation is in t1tables.h */

  FT_EXPORT_DEF( FT_Int )
  FT_Has_PS_Glyph_Names( FT_Face  face )
  {
    FT_Int             result = 0;
    FT_Service_PsInfo  service;


    if ( face )
    {
      FT_FACE_FIND_SERVICE( face, service, POSTSCRIPT_INFO );

      if ( service && service->ps_has_glyph_names )
        result = service->ps_has_glyph_names( face );
    }

    return result;
  }


  /* documentation is in t1tables.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_PS_Font_Private( FT_Face         face,
                          PS_PrivateRec*  afont_private )
  {
    FT_Error           error;
    FT_Service_PsInfo  service;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( !afont_private )
      return FT_THROW( Invalid_Argument );

    FT_FACE_FIND_SERVICE( face, service, POSTSCRIPT_INFO );

    if ( service && service->ps_get_font_private )
      error = service->ps_get_font_private( face, afont_private );
    else
      error = FT_THROW( Invalid_Argument );

    return error;
  }


  /* documentation is in t1tables.h */

  FT_EXPORT_DEF( FT_Long )
  FT_Get_PS_Font_Value( FT_Face       face,
                        PS_Dict_Keys  key,
                        FT_UInt       idx,
                        void         *value,
                        FT_Long       value_len )
  {
    FT_Int             result  = 0;
    FT_Service_PsInfo  service = NULL;


    if ( face )
    {
      FT_FACE_FIND_SERVICE( face, service, POSTSCRIPT_INFO );

      if ( service && service->ps_get_font_value )
        result = service->ps_get_font_value( face, key, idx,
                                             value, value_len );
    }

    return result;
  }


/* END */
