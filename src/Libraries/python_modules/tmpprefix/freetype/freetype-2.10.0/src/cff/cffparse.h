/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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
#ifndef CFFPARSE_H_
#define CFFPARSE_H_


#include <ft2build.h>
#include FT_INTERNAL_CFF_TYPES_H
#include FT_INTERNAL_OBJECTS_H


FT_BEGIN_HEADER


  /* CFF uses constant parser stack size; */
  /* CFF2 can increase from default 193   */
#define CFF_MAX_STACK_DEPTH  96

  /*
   * There are plans to remove the `maxstack' operator in a forthcoming
   * revision of the CFF2 specification, increasing the (then static) stack
   * size to 513.  By making the default stack size equal to the maximum
   * stack size, the operator is essentially disabled, which has the
   * desired effect in FreeType.
   */
#define CFF2_MAX_STACK      513
#define CFF2_DEFAULT_STACK  513

#define CFF_CODE_TOPDICT    0x1000
#define CFF_CODE_PRIVATE    0x2000
#define CFF2_CODE_TOPDICT   0x3000
#define CFF2_CODE_FONTDICT  0x4000
#define CFF2_CODE_PRIVATE   0x5000


  typedef struct  CFF_ParserRec_
  {
    FT_Library  library;
    FT_Byte*    start;
    FT_Byte*    limit;
    FT_Byte*    cursor;

    FT_Byte**   stack;
    FT_Byte**   top;
    FT_UInt     stackSize;  /* allocated size */

    FT_UInt     object_code;
    void*       object;

    FT_UShort   num_designs; /* a copy of `CFF_FontRecDict->num_designs' */
    FT_UShort   num_axes;    /* a copy of `CFF_FontRecDict->num_axes'    */

  } CFF_ParserRec, *CFF_Parser;


  FT_LOCAL( FT_Long )
  cff_parse_num( CFF_Parser  parser,
                 FT_Byte**   d );

  FT_LOCAL( FT_Error )
  cff_parser_init( CFF_Parser  parser,
                   FT_UInt     code,
                   void*       object,
                   FT_Library  library,
                   FT_UInt     stackSize,
                   FT_UShort   num_designs,
                   FT_UShort   num_axes );

  FT_LOCAL( void )
  cff_parser_done( CFF_Parser  parser );

  FT_LOCAL( FT_Error )
  cff_parser_run( CFF_Parser  parser,
                  FT_Byte*    start,
                  FT_Byte*    limit );


  enum
  {
    cff_kind_none = 0,
    cff_kind_num,
    cff_kind_fixed,
    cff_kind_fixed_thousand,
    cff_kind_string,
    cff_kind_bool,
    cff_kind_delta,
    cff_kind_callback,
    cff_kind_blend,

    cff_kind_max  /* do not remove */
  };


  /* now generate handlers for the most simple fields */
  typedef FT_Error  (*CFF_Field_Reader)( CFF_Parser  parser );

  typedef struct  CFF_Field_Handler_
  {
    int               kind;
    int               code;
    FT_UInt           offset;
    FT_Byte           size;
    CFF_Field_Reader  reader;
    FT_UInt           array_max;
    FT_UInt           count_offset;

#ifdef FT_DEBUG_LEVEL_TRACE
    const char*       id;
#endif

  } CFF_Field_Handler;


FT_END_HEADER


#endif /* CFFPARSE_H_ */


/* END */
