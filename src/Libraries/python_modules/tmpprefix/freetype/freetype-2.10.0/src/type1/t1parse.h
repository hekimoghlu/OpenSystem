/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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
#ifndef T1PARSE_H_
#define T1PARSE_H_


#include <ft2build.h>
#include FT_INTERNAL_TYPE1_TYPES_H
#include FT_INTERNAL_STREAM_H


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @Struct:
   *   T1_ParserRec
   *
   * @Description:
   *   A PS_ParserRec is an object used to parse a Type 1 fonts very
   *   quickly.
   *
   * @Fields:
   *   root ::
   *     The root parser.
   *
   *   stream ::
   *     The current input stream.
   *
   *   base_dict ::
   *     A pointer to the top-level dictionary.
   *
   *   base_len ::
   *     The length in bytes of the top dictionary.
   *
   *   private_dict ::
   *     A pointer to the private dictionary.
   *
   *   private_len ::
   *     The length in bytes of the private dictionary.
   *
   *   in_pfb ::
   *     A boolean.  Indicates that we are handling a PFB
   *     file.
   *
   *   in_memory ::
   *     A boolean.  Indicates a memory-based stream.
   *
   *   single_block ::
   *     A boolean.  Indicates that the private dictionary
   *     is stored in lieu of the base dictionary.
   */
  typedef struct  T1_ParserRec_
  {
    PS_ParserRec  root;
    FT_Stream     stream;

    FT_Byte*      base_dict;
    FT_ULong      base_len;

    FT_Byte*      private_dict;
    FT_ULong      private_len;

    FT_Bool       in_pfb;
    FT_Bool       in_memory;
    FT_Bool       single_block;

  } T1_ParserRec, *T1_Parser;


#define T1_Add_Table( p, i, o, l )  (p)->funcs.add( (p), i, o, l )
#define T1_Release_Table( p )          \
          do                           \
          {                            \
            if ( (p)->funcs.release )  \
              (p)->funcs.release( p ); \
          } while ( 0 )


#define T1_Skip_Spaces( p )    (p)->root.funcs.skip_spaces( &(p)->root )
#define T1_Skip_PS_Token( p )  (p)->root.funcs.skip_PS_token( &(p)->root )

#define T1_ToInt( p )       (p)->root.funcs.to_int( &(p)->root )
#define T1_ToFixed( p, t )  (p)->root.funcs.to_fixed( &(p)->root, t )

#define T1_ToCoordArray( p, m, c )                           \
          (p)->root.funcs.to_coord_array( &(p)->root, m, c )
#define T1_ToFixedArray( p, m, f, t )                           \
          (p)->root.funcs.to_fixed_array( &(p)->root, m, f, t )
#define T1_ToToken( p, t )                          \
          (p)->root.funcs.to_token( &(p)->root, t )
#define T1_ToTokenArray( p, t, m, c )                           \
          (p)->root.funcs.to_token_array( &(p)->root, t, m, c )

#define T1_Load_Field( p, f, o, m, pf )                         \
          (p)->root.funcs.load_field( &(p)->root, f, o, m, pf )

#define T1_Load_Field_Table( p, f, o, m, pf )                         \
          (p)->root.funcs.load_field_table( &(p)->root, f, o, m, pf )


  FT_LOCAL( FT_Error )
  T1_New_Parser( T1_Parser      parser,
                 FT_Stream      stream,
                 FT_Memory      memory,
                 PSAux_Service  psaux );

  FT_LOCAL( FT_Error )
  T1_Get_Private_Dict( T1_Parser      parser,
                       PSAux_Service  psaux );

  FT_LOCAL( void )
  T1_Finalize_Parser( T1_Parser  parser );


FT_END_HEADER

#endif /* T1PARSE_H_ */


/* END */
