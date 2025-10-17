/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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
#ifndef SVPSCMAP_H_
#define SVPSCMAP_H_

#include FT_INTERNAL_OBJECTS_H


FT_BEGIN_HEADER


#define FT_SERVICE_ID_POSTSCRIPT_CMAPS  "postscript-cmaps"


  /*
   * Adobe glyph name to unicode value.
   */
  typedef FT_UInt32
  (*PS_Unicode_ValueFunc)( const char*  glyph_name );

  /*
   * Macintosh name id to glyph name.  `NULL` if invalid index.
   */
  typedef const char*
  (*PS_Macintosh_NameFunc)( FT_UInt  name_index );

  /*
   * Adobe standard string ID to glyph name.  `NULL` if invalid index.
   */
  typedef const char*
  (*PS_Adobe_Std_StringsFunc)( FT_UInt  string_index );


  /*
   * Simple unicode -> glyph index charmap built from font glyph names table.
   */
  typedef struct  PS_UniMap_
  {
    FT_UInt32  unicode;      /* bit 31 set: is glyph variant */
    FT_UInt    glyph_index;

  } PS_UniMap;


  typedef struct PS_UnicodesRec_*  PS_Unicodes;

  typedef struct  PS_UnicodesRec_
  {
    FT_CMapRec  cmap;
    FT_UInt     num_maps;
    PS_UniMap*  maps;

  } PS_UnicodesRec;


  /*
   * A function which returns a glyph name for a given index.  Returns
   * `NULL` if invalid index.
   */
  typedef const char*
  (*PS_GetGlyphNameFunc)( FT_Pointer  data,
                          FT_UInt     string_index );

  /*
   * A function used to release the glyph name returned by
   * PS_GetGlyphNameFunc, when needed
   */
  typedef void
  (*PS_FreeGlyphNameFunc)( FT_Pointer  data,
                           const char*  name );

  typedef FT_Error
  (*PS_Unicodes_InitFunc)( FT_Memory             memory,
                           PS_Unicodes           unicodes,
                           FT_UInt               num_glyphs,
                           PS_GetGlyphNameFunc   get_glyph_name,
                           PS_FreeGlyphNameFunc  free_glyph_name,
                           FT_Pointer            glyph_data );

  typedef FT_UInt
  (*PS_Unicodes_CharIndexFunc)( PS_Unicodes  unicodes,
                                FT_UInt32    unicode );

  typedef FT_UInt32
  (*PS_Unicodes_CharNextFunc)( PS_Unicodes  unicodes,
                               FT_UInt32   *unicode );


  FT_DEFINE_SERVICE( PsCMaps )
  {
    PS_Unicode_ValueFunc       unicode_value;

    PS_Unicodes_InitFunc       unicodes_init;
    PS_Unicodes_CharIndexFunc  unicodes_char_index;
    PS_Unicodes_CharNextFunc   unicodes_char_next;

    PS_Macintosh_NameFunc      macintosh_name;
    PS_Adobe_Std_StringsFunc   adobe_std_strings;
    const unsigned short*      adobe_std_encoding;
    const unsigned short*      adobe_expert_encoding;
  };


#define FT_DEFINE_SERVICE_PSCMAPSREC( class_,                               \
                                      unicode_value_,                       \
                                      unicodes_init_,                       \
                                      unicodes_char_index_,                 \
                                      unicodes_char_next_,                  \
                                      macintosh_name_,                      \
                                      adobe_std_strings_,                   \
                                      adobe_std_encoding_,                  \
                                      adobe_expert_encoding_ )              \
  static const FT_Service_PsCMapsRec  class_ =                              \
  {                                                                         \
    unicode_value_, unicodes_init_,                                         \
    unicodes_char_index_, unicodes_char_next_, macintosh_name_,             \
    adobe_std_strings_, adobe_std_encoding_, adobe_expert_encoding_         \
  };

  /* */


FT_END_HEADER


#endif /* SVPSCMAP_H_ */


/* END */
