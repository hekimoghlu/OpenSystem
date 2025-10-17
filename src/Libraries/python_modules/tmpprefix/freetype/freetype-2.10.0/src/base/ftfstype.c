/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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
#include FT_TYPE1_TABLES_H
#include FT_TRUETYPE_TABLES_H
#include FT_INTERNAL_SERVICE_H
#include FT_SERVICE_POSTSCRIPT_INFO_H


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_UShort )
  FT_Get_FSType_Flags( FT_Face  face )
  {
    TT_OS2*  os2;


    /* first, try to get the fs_type directly from the font */
    if ( face )
    {
      FT_Service_PsInfo  service = NULL;


      FT_FACE_FIND_SERVICE( face, service, POSTSCRIPT_INFO );

      if ( service && service->ps_get_font_extra )
      {
        PS_FontExtraRec  extra;


        if ( !service->ps_get_font_extra( face, &extra ) &&
             extra.fs_type != 0                          )
          return extra.fs_type;
      }
    }

    /* look at FSType before fsType for Type42 */

    if ( ( os2 = (TT_OS2*)FT_Get_Sfnt_Table( face, FT_SFNT_OS2 ) ) != NULL &&
         os2->version != 0xFFFFU                                           )
      return os2->fsType;

    return 0;
  }


/* END */
