/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 7, 2022.
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
#ifndef T1LOAD_H_
#define T1LOAD_H_


#include <ft2build.h>
#include FT_INTERNAL_STREAM_H
#include FT_INTERNAL_POSTSCRIPT_AUX_H
#include FT_MULTIPLE_MASTERS_H

#include "t1parse.h"


FT_BEGIN_HEADER


  typedef struct  T1_Loader_
  {
    T1_ParserRec  parser;          /* parser used to read the stream */

    FT_Int        num_chars;       /* number of characters in encoding */
    PS_TableRec   encoding_table;  /* PS_Table used to store the       */
                                   /* encoding character names         */

    FT_Int        num_glyphs;
    PS_TableRec   glyph_names;
    PS_TableRec   charstrings;
    PS_TableRec   swap_table;      /* For moving .notdef glyph to index 0. */

    FT_Int        num_subrs;
    PS_TableRec   subrs;
    FT_Hash       subrs_hash;
    FT_Bool       fontdata;

    FT_UInt       keywords_encountered; /* T1_LOADER_ENCOUNTERED_XXX */

  } T1_LoaderRec, *T1_Loader;


  /* treatment of some keywords differs depending on whether */
  /* they precede or follow certain other keywords           */

#define T1_PRIVATE                ( 1 << 0 )
#define T1_FONTDIR_AFTER_PRIVATE  ( 1 << 1 )


  FT_LOCAL( FT_Error )
  T1_Open_Face( T1_Face  face );

#ifndef T1_CONFIG_OPTION_NO_MM_SUPPORT

  FT_LOCAL( FT_Error )
  T1_Get_Multi_Master( T1_Face           face,
                       FT_Multi_Master*  master );

  FT_LOCAL( FT_Error )
  T1_Get_MM_Var( T1_Face      face,
                 FT_MM_Var*  *master );

  FT_LOCAL( FT_Error )
  T1_Set_MM_Blend( T1_Face    face,
                   FT_UInt    num_coords,
                   FT_Fixed*  coords );

  FT_LOCAL( FT_Error )
  T1_Get_MM_Blend( T1_Face    face,
                   FT_UInt    num_coords,
                   FT_Fixed*  coords );

  FT_LOCAL( FT_Error )
  T1_Set_MM_Design( T1_Face   face,
                    FT_UInt   num_coords,
                    FT_Long*  coords );

  FT_LOCAL( FT_Error )
  T1_Reset_MM_Blend( T1_Face  face,
                     FT_UInt  instance_index );

  FT_LOCAL( FT_Error )
  T1_Get_Var_Design( T1_Face    face,
                     FT_UInt    num_coords,
                     FT_Fixed*  coords );

  FT_LOCAL( FT_Error )
  T1_Set_Var_Design( T1_Face    face,
                     FT_UInt    num_coords,
                     FT_Fixed*  coords );

  FT_LOCAL( void )
  T1_Done_Blend( T1_Face  face );

  FT_LOCAL( FT_Error )
  T1_Set_MM_WeightVector( T1_Face    face,
                          FT_UInt    len,
                          FT_Fixed*  weightvector );

  FT_LOCAL( FT_Error )
  T1_Get_MM_WeightVector( T1_Face    face,
                          FT_UInt*   len,
                          FT_Fixed*  weightvector );

#endif /* !T1_CONFIG_OPTION_NO_MM_SUPPORT */


FT_END_HEADER

#endif /* T1LOAD_H_ */


/* END */
