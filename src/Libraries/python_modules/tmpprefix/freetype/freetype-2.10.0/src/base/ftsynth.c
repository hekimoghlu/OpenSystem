/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 21, 2024.
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
#include <ft2build.h>
#include FT_SYNTHESIS_H
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_OBJECTS_H
#include FT_OUTLINE_H
#include FT_BITMAP_H


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  synth


  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****   EXPERIMENTAL OBLIQUING SUPPORT                                ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/

  /* documentation is in ftsynth.h */

  FT_EXPORT_DEF( void )
  FT_GlyphSlot_Oblique( FT_GlyphSlot  slot )
  {
    FT_Matrix    transform;
    FT_Outline*  outline;


    if ( !slot )
      return;

    outline = &slot->outline;

    /* only oblique outline glyphs */
    if ( slot->format != FT_GLYPH_FORMAT_OUTLINE )
      return;

    /* we don't touch the advance width */

    /* For italic, simply apply a shear transform, with an angle */
    /* of about 12 degrees.                                      */

    transform.xx = 0x10000L;
    transform.yx = 0x00000L;

    transform.xy = 0x0366AL;
    transform.yy = 0x10000L;

    FT_Outline_Transform( outline, &transform );
  }


  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****   EXPERIMENTAL EMBOLDENING SUPPORT                              ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/


  /* documentation is in ftsynth.h */

  FT_EXPORT_DEF( void )
  FT_GlyphSlot_Embolden( FT_GlyphSlot  slot )
  {
    FT_Library  library;
    FT_Face     face;
    FT_Error    error;
    FT_Pos      xstr, ystr;


    if ( !slot )
      return;

    library = slot->library;
    face    = slot->face;

    if ( slot->format != FT_GLYPH_FORMAT_OUTLINE &&
         slot->format != FT_GLYPH_FORMAT_BITMAP  )
      return;

    /* some reasonable strength */
    xstr = FT_MulFix( face->units_per_EM,
                      face->size->metrics.y_scale ) / 24;
    ystr = xstr;

    if ( slot->format == FT_GLYPH_FORMAT_OUTLINE )
      FT_Outline_EmboldenXY( &slot->outline, xstr, ystr );

    else /* slot->format == FT_GLYPH_FORMAT_BITMAP */
    {
      /* round to full pixels */
      xstr &= ~63;
      if ( xstr == 0 )
        xstr = 1 << 6;
      ystr &= ~63;

      /*
       * XXX: overflow check for 16-bit system, for compatibility
       *      with FT_GlyphSlot_Embolden() since FreeType 2.1.10.
       *      unfortunately, this function return no informations
       *      about the cause of error.
       */
      if ( ( ystr >> 6 ) > FT_INT_MAX || ( ystr >> 6 ) < FT_INT_MIN )
      {
        FT_TRACE1(( "FT_GlyphSlot_Embolden:" ));
        FT_TRACE1(( "too strong emboldening parameter ystr=%d\n", ystr ));
        return;
      }
      error = FT_GlyphSlot_Own_Bitmap( slot );
      if ( error )
        return;

      error = FT_Bitmap_Embolden( library, &slot->bitmap, xstr, ystr );
      if ( error )
        return;
    }

    if ( slot->advance.x )
      slot->advance.x += xstr;

    if ( slot->advance.y )
      slot->advance.y += ystr;

    slot->metrics.width        += xstr;
    slot->metrics.height       += ystr;
    slot->metrics.horiAdvance  += xstr;
    slot->metrics.vertAdvance  += ystr;
    slot->metrics.horiBearingY += ystr;

    /* XXX: 16-bit overflow case must be excluded before here */
    if ( slot->format == FT_GLYPH_FORMAT_BITMAP )
      slot->bitmap_top += (FT_Int)( ystr >> 6 );
  }


/* END */
