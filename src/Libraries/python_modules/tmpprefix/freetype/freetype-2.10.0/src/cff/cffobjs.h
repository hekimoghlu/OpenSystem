/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 5, 2024.
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
#ifndef CFFOBJS_H_
#define CFFOBJS_H_


#include <ft2build.h>


FT_BEGIN_HEADER


  FT_LOCAL( FT_Error )
  cff_size_init( FT_Size  size );           /* CFF_Size */

  FT_LOCAL( void )
  cff_size_done( FT_Size  size );           /* CFF_Size */

  FT_LOCAL( FT_Error )
  cff_size_request( FT_Size          size,
                    FT_Size_Request  req );

#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

  FT_LOCAL( FT_Error )
  cff_size_select( FT_Size   size,
                   FT_ULong  strike_index );

#endif

  FT_LOCAL( void )
  cff_slot_done( FT_GlyphSlot  slot );

  FT_LOCAL( FT_Error )
  cff_slot_init( FT_GlyphSlot  slot );


  /**************************************************************************
   *
   * Face functions
   */
  FT_LOCAL( FT_Error )
  cff_face_init( FT_Stream      stream,
                 FT_Face        face,           /* CFF_Face */
                 FT_Int         face_index,
                 FT_Int         num_params,
                 FT_Parameter*  params );

  FT_LOCAL( void )
  cff_face_done( FT_Face  face );               /* CFF_Face */


  /**************************************************************************
   *
   * Driver functions
   */
  FT_LOCAL( FT_Error )
  cff_driver_init( FT_Module  module );         /* PS_Driver */

  FT_LOCAL( void )
  cff_driver_done( FT_Module  module );         /* PS_Driver */


FT_END_HEADER

#endif /* CFFOBJS_H_ */


/* END */
