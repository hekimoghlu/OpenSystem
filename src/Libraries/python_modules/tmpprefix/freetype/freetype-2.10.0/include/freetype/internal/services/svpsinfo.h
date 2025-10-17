/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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
#ifndef SVPSINFO_H_
#define SVPSINFO_H_

#include FT_INTERNAL_SERVICE_H
#include FT_INTERNAL_TYPE1_TYPES_H


FT_BEGIN_HEADER


#define FT_SERVICE_ID_POSTSCRIPT_INFO  "postscript-info"


  typedef FT_Error
  (*PS_GetFontInfoFunc)( FT_Face          face,
                         PS_FontInfoRec*  afont_info );

  typedef FT_Error
  (*PS_GetFontExtraFunc)( FT_Face           face,
                          PS_FontExtraRec*  afont_extra );

  typedef FT_Int
  (*PS_HasGlyphNamesFunc)( FT_Face  face );

  typedef FT_Error
  (*PS_GetFontPrivateFunc)( FT_Face         face,
                            PS_PrivateRec*  afont_private );

  typedef FT_Long
  (*PS_GetFontValueFunc)( FT_Face       face,
                          PS_Dict_Keys  key,
                          FT_UInt       idx,
                          void         *value,
                          FT_Long       value_len );


  FT_DEFINE_SERVICE( PsInfo )
  {
    PS_GetFontInfoFunc     ps_get_font_info;
    PS_GetFontExtraFunc    ps_get_font_extra;
    PS_HasGlyphNamesFunc   ps_has_glyph_names;
    PS_GetFontPrivateFunc  ps_get_font_private;
    PS_GetFontValueFunc    ps_get_font_value;
  };


#define FT_DEFINE_SERVICE_PSINFOREC( class_,                     \
                                     get_font_info_,             \
                                     ps_get_font_extra_,         \
                                     has_glyph_names_,           \
                                     get_font_private_,          \
                                     get_font_value_ )           \
  static const FT_Service_PsInfoRec  class_ =                    \
  {                                                              \
    get_font_info_, ps_get_font_extra_, has_glyph_names_,        \
    get_font_private_, get_font_value_                           \
  };

  /* */


FT_END_HEADER


#endif /* SVPSINFO_H_ */


/* END */
