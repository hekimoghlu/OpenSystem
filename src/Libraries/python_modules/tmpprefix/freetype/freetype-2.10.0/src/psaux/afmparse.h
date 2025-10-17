/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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
#ifndef AFMPARSE_H_
#define AFMPARSE_H_


#include <ft2build.h>
#include FT_INTERNAL_POSTSCRIPT_AUX_H


FT_BEGIN_HEADER


  FT_LOCAL( FT_Error )
  afm_parser_init( AFM_Parser  parser,
                   FT_Memory   memory,
                   FT_Byte*    base,
                   FT_Byte*    limit );


  FT_LOCAL( void )
  afm_parser_done( AFM_Parser  parser );


  FT_LOCAL( FT_Error )
  afm_parser_parse( AFM_Parser  parser );


  enum  AFM_ValueType_
  {
    AFM_VALUE_TYPE_STRING,
    AFM_VALUE_TYPE_NAME,
    AFM_VALUE_TYPE_FIXED,   /* real number */
    AFM_VALUE_TYPE_INTEGER,
    AFM_VALUE_TYPE_BOOL,
    AFM_VALUE_TYPE_INDEX    /* glyph index */
  };


  typedef struct  AFM_ValueRec_
  {
    enum AFM_ValueType_  type;
    union
    {
      char*     s;
      FT_Fixed  f;
      FT_Int    i;
      FT_UInt   u;
      FT_Bool   b;

    } u;

  } AFM_ValueRec, *AFM_Value;

#define  AFM_MAX_ARGUMENTS  5

  FT_LOCAL( FT_Int )
  afm_parser_read_vals( AFM_Parser  parser,
                        AFM_Value   vals,
                        FT_Int      n );

  /* read the next key from the next line or column */
  FT_LOCAL( char* )
  afm_parser_next_key( AFM_Parser  parser,
                       FT_Bool     line,
                       FT_Offset*  len );

FT_END_HEADER

#endif /* AFMPARSE_H_ */


/* END */
