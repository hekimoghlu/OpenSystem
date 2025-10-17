/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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
#ifndef T42OBJS_H_
#define T42OBJS_H_

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_TYPE1_TABLES_H
#include FT_INTERNAL_TYPE1_TYPES_H
#include "t42types.h"
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_DRIVER_H
#include FT_SERVICE_POSTSCRIPT_CMAPS_H
#include FT_INTERNAL_POSTSCRIPT_HINTS_H


FT_BEGIN_HEADER


  /* Type42 size */
  typedef struct  T42_SizeRec_
  {
    FT_SizeRec  root;
    FT_Size     ttsize;

  } T42_SizeRec, *T42_Size;


  /* Type42 slot */
  typedef struct  T42_GlyphSlotRec_
  {
    FT_GlyphSlotRec  root;
    FT_GlyphSlot     ttslot;

  } T42_GlyphSlotRec, *T42_GlyphSlot;


  /* Type 42 driver */
  typedef struct  T42_DriverRec_
  {
    FT_DriverRec     root;
    FT_Driver_Class  ttclazz;

  } T42_DriverRec, *T42_Driver;


  /* */


  FT_LOCAL( FT_Error )
  T42_Face_Init( FT_Stream      stream,
                 FT_Face        face,
                 FT_Int         face_index,
                 FT_Int         num_params,
                 FT_Parameter*  params );


  FT_LOCAL( void )
  T42_Face_Done( FT_Face  face );


  FT_LOCAL( FT_Error )
  T42_Size_Init( FT_Size  size );


  FT_LOCAL( FT_Error )
  T42_Size_Request( FT_Size          size,
                    FT_Size_Request  req );


  FT_LOCAL( FT_Error )
  T42_Size_Select( FT_Size   size,
                   FT_ULong  strike_index );


  FT_LOCAL( void )
  T42_Size_Done( FT_Size  size );


  FT_LOCAL( FT_Error )
  T42_GlyphSlot_Init( FT_GlyphSlot  slot );


  FT_LOCAL( FT_Error )
  T42_GlyphSlot_Load( FT_GlyphSlot  glyph,
                      FT_Size       size,
                      FT_UInt       glyph_index,
                      FT_Int32      load_flags );

  FT_LOCAL( void )
  T42_GlyphSlot_Done( FT_GlyphSlot  slot );


  FT_LOCAL( FT_Error )
  T42_Driver_Init( FT_Module  module );

  FT_LOCAL( void )
  T42_Driver_Done( FT_Module  module );

 /* */

FT_END_HEADER


#endif /* T42OBJS_H_ */


/* END */
