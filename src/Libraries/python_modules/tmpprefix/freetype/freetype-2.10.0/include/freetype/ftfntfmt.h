/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
#ifndef FTFNTFMT_H_
#define FTFNTFMT_H_

#include <ft2build.h>
#include FT_FREETYPE_H

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @section:
   *  font_formats
   *
   * @title:
   *  Font Formats
   *
   * @abstract:
   *  Getting the font format.
   *
   * @description:
   *  The single function in this section can be used to get the font format.
   *  Note that this information is not needed normally; however, there are
   *  special cases (like in PDF devices) where it is important to
   *  differentiate, in spite of FreeType's uniform API.
   *
   */


  /**************************************************************************
   *
   * @function:
   *  FT_Get_Font_Format
   *
   * @description:
   *  Return a string describing the format of a given face.  Possible values
   *  are 'TrueType', 'Type~1', 'BDF', 'PCF', 'Type~42', 'CID~Type~1', 'CFF',
   *  'PFR', and 'Windows~FNT'.
   *
   *  The return value is suitable to be used as an X11 FONT_PROPERTY.
   *
   * @input:
   *  face ::
   *    Input face handle.
   *
   * @return:
   *  Font format string.  `NULL` in case of error.
   *
   * @note:
   *  A deprecated name for the same function is `FT_Get_X11_Font_Format`.
   */
  FT_EXPORT( const char* )
  FT_Get_Font_Format( FT_Face  face );


  /* deprecated */
  FT_EXPORT( const char* )
  FT_Get_X11_Font_Format( FT_Face  face );


  /* */


FT_END_HEADER

#endif /* FTFNTFMT_H_ */


/* END */
