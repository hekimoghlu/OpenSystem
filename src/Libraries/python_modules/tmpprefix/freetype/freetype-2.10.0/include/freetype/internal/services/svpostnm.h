/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#ifndef SVPOSTNM_H_
#define SVPOSTNM_H_

#include FT_INTERNAL_SERVICE_H


FT_BEGIN_HEADER

  /*
   * A trivial service used to retrieve the PostScript name of a given font
   * when available.  The `get_name' field should never be `NULL`.
   *
   * The corresponding function can return `NULL` to indicate that the
   * PostScript name is not available.
   *
   * The name is owned by the face and will be destroyed with it.
   */

#define FT_SERVICE_ID_POSTSCRIPT_FONT_NAME  "postscript-font-name"


  typedef const char*
  (*FT_PsName_GetFunc)( FT_Face  face );


  FT_DEFINE_SERVICE( PsFontName )
  {
    FT_PsName_GetFunc  get_ps_font_name;
  };


#define FT_DEFINE_SERVICE_PSFONTNAMEREC( class_, get_ps_font_name_ ) \
  static const FT_Service_PsFontNameRec  class_ =                    \
  {                                                                  \
    get_ps_font_name_                                                \
  };

  /* */


FT_END_HEADER


#endif /* SVPOSTNM_H_ */


/* END */
