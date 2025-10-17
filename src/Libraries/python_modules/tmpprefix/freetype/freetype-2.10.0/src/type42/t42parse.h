/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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
#ifndef T42PARSE_H_
#define T42PARSE_H_


#include "t42objs.h"
#include FT_INTERNAL_POSTSCRIPT_AUX_H


FT_BEGIN_HEADER

  typedef struct  T42_ParserRec_
  {
    PS_ParserRec  root;
    FT_Stream     stream;

    FT_Byte*      base_dict;
    FT_Long       base_len;

    FT_Bool       in_memory;

  } T42_ParserRec, *T42_Parser;


  typedef struct  T42_Loader_
  {
    T42_ParserRec  parser;          /* parser used to read the stream */

    FT_Int         num_chars;       /* number of characters in encoding */
    PS_TableRec    encoding_table;  /* PS_Table used to store the       */
                                    /* encoding character names         */

    FT_Int         num_glyphs;
    PS_TableRec    glyph_names;
    PS_TableRec    charstrings;
    PS_TableRec    swap_table;      /* For moving .notdef glyph to index 0. */

  } T42_LoaderRec, *T42_Loader;


  FT_LOCAL( FT_Error )
  t42_parser_init( T42_Parser     parser,
                   FT_Stream      stream,
                   FT_Memory      memory,
                   PSAux_Service  psaux );

  FT_LOCAL( void )
  t42_parser_done( T42_Parser  parser );


  FT_LOCAL( FT_Error )
  t42_parse_dict( T42_Face    face,
                  T42_Loader  loader,
                  FT_Byte*    base,
                  FT_Long     size );


  FT_LOCAL( void )
  t42_loader_init( T42_Loader  loader,
                   T42_Face    face );

  FT_LOCAL( void )
  t42_loader_done( T42_Loader  loader );


 /* */

FT_END_HEADER


#endif /* T42PARSE_H_ */


/* END */
