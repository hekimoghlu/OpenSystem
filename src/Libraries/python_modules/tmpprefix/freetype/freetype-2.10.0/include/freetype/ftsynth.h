/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
/*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*********                                                       *********/
  /*********        WARNING, THIS IS ALPHA CODE!  THIS API         *********/
  /*********    IS DUE TO CHANGE UNTIL STRICTLY NOTIFIED BY THE    *********/
  /*********            FREETYPE DEVELOPMENT TEAM                  *********/
  /*********                                                       *********/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


  /* Main reason for not lifting the functions in this module to a  */
  /* 'standard' API is that the used parameters for emboldening and */
  /* slanting are not configurable.  Consider the functions as a    */
  /* code resource that should be copied into the application and   */
  /* adapted to the particular needs.                               */


#ifndef FTSYNTH_H_
#define FTSYNTH_H_


#include <ft2build.h>
#include FT_FREETYPE_H

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER

  /* Embolden a glyph by a 'reasonable' value (which is highly a matter of */
  /* taste).  This function is actually a convenience function, providing  */
  /* a wrapper for @FT_Outline_Embolden and @FT_Bitmap_Embolden.           */
  /*                                                                       */
  /* For emboldened outlines the height, width, and advance metrics are    */
  /* increased by the strength of the emboldening -- this even affects     */
  /* mono-width fonts!                                                     */
  /*                                                                       */
  /* You can also call @FT_Outline_Get_CBox to get precise values.         */
  FT_EXPORT( void )
  FT_GlyphSlot_Embolden( FT_GlyphSlot  slot );

  /* Slant an outline glyph to the right by about 12 degrees. */
  FT_EXPORT( void )
  FT_GlyphSlot_Oblique( FT_GlyphSlot  slot );

  /* */


FT_END_HEADER

#endif /* FTSYNTH_H_ */


/* END */
