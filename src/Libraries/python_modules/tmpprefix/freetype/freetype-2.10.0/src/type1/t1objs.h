/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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
#ifndef T1OBJS_H_
#define T1OBJS_H_


#include <ft2build.h>
#include FT_INTERNAL_OBJECTS_H
#include FT_CONFIG_CONFIG_H
#include FT_INTERNAL_TYPE1_TYPES_H


FT_BEGIN_HEADER


  /* The following structures must be defined by the hinter */
  typedef struct T1_Size_Hints_   T1_Size_Hints;
  typedef struct T1_Glyph_Hints_  T1_Glyph_Hints;


  /**************************************************************************
   *
   * @Type:
   *   T1_Size
   *
   * @Description:
   *   A handle to a Type 1 size object.
   */
  typedef struct T1_SizeRec_*  T1_Size;


  /**************************************************************************
   *
   * @Type:
   *   T1_GlyphSlot
   *
   * @Description:
   *   A handle to a Type 1 glyph slot object.
   */
  typedef struct T1_GlyphSlotRec_*  T1_GlyphSlot;


  /**************************************************************************
   *
   * @Type:
   *   T1_CharMap
   *
   * @Description:
   *   A handle to a Type 1 character mapping object.
   *
   * @Note:
   *   The Type 1 format doesn't use a charmap but an encoding table.
   *   The driver is responsible for making up charmap objects
   *   corresponding to these tables.
   */
  typedef struct T1_CharMapRec_*   T1_CharMap;


  /**************************************************************************
   *
   *                 HERE BEGINS THE TYPE1 SPECIFIC STUFF
   *
   */


  /**************************************************************************
   *
   * @Type:
   *   T1_SizeRec
   *
   * @Description:
   *   Type 1 size record.
   */
  typedef struct  T1_SizeRec_
  {
    FT_SizeRec  root;

  } T1_SizeRec;


  FT_LOCAL( void )
  T1_Size_Done( FT_Size  size );

  FT_LOCAL( FT_Error )
  T1_Size_Request( FT_Size          size,
                   FT_Size_Request  req );

  FT_LOCAL( FT_Error )
  T1_Size_Init( FT_Size  size );


  /**************************************************************************
   *
   * @Type:
   *   T1_GlyphSlotRec
   *
   * @Description:
   *   Type 1 glyph slot record.
   */
  typedef struct  T1_GlyphSlotRec_
  {
    FT_GlyphSlotRec  root;

    FT_Bool          hint;
    FT_Bool          scaled;

    FT_Fixed         x_scale;
    FT_Fixed         y_scale;

    FT_Int           max_points;
    FT_Int           max_contours;

  } T1_GlyphSlotRec;


  FT_LOCAL( FT_Error )
  T1_Face_Init( FT_Stream      stream,
                FT_Face        face,
                FT_Int         face_index,
                FT_Int         num_params,
                FT_Parameter*  params );

  FT_LOCAL( void )
  T1_Face_Done( FT_Face  face );

  FT_LOCAL( FT_Error )
  T1_GlyphSlot_Init( FT_GlyphSlot  slot );

  FT_LOCAL( void )
  T1_GlyphSlot_Done( FT_GlyphSlot  slot );

  FT_LOCAL( FT_Error )
  T1_Driver_Init( FT_Module  driver );

  FT_LOCAL( void )
  T1_Driver_Done( FT_Module  driver );


FT_END_HEADER

#endif /* T1OBJS_H_ */


/* END */
