/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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
#ifndef PFROBJS_H_
#define PFROBJS_H_

#include "pfrtypes.h"


FT_BEGIN_HEADER

  typedef struct PFR_FaceRec_*  PFR_Face;

  typedef struct PFR_SizeRec_*  PFR_Size;

  typedef struct PFR_SlotRec_*  PFR_Slot;


  typedef struct  PFR_FaceRec_
  {
    FT_FaceRec      root;
    PFR_HeaderRec   header;
    PFR_LogFontRec  log_font;
    PFR_PhyFontRec  phy_font;

  } PFR_FaceRec;


  typedef struct  PFR_SizeRec_
  {
    FT_SizeRec  root;

  } PFR_SizeRec;


  typedef struct  PFR_SlotRec_
  {
    FT_GlyphSlotRec  root;
    PFR_GlyphRec     glyph;

  } PFR_SlotRec;


  FT_LOCAL( FT_Error )
  pfr_face_init( FT_Stream      stream,
                 FT_Face        face,           /* PFR_Face */
                 FT_Int         face_index,
                 FT_Int         num_params,
                 FT_Parameter*  params );

  FT_LOCAL( void )
  pfr_face_done( FT_Face  face );               /* PFR_Face */


  FT_LOCAL( FT_Error )
  pfr_face_get_kerning( FT_Face     face,       /* PFR_Face */
                        FT_UInt     glyph1,
                        FT_UInt     glyph2,
                        FT_Vector*  kerning );


  FT_LOCAL( FT_Error )
  pfr_slot_init( FT_GlyphSlot  slot );          /* PFR_Slot */

  FT_LOCAL( void )
  pfr_slot_done( FT_GlyphSlot  slot );          /* PFR_Slot */


  FT_LOCAL( FT_Error )
  pfr_slot_load( FT_GlyphSlot  slot,            /* PFR_Slot */
                 FT_Size       size,            /* PFR_Size */
                 FT_UInt       gindex,
                 FT_Int32      load_flags );


FT_END_HEADER

#endif /* PFROBJS_H_ */


/* END */
