/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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
#include FT_INTERNAL_DEBUG_H

#include FT_ADVANCES_H
#include FT_INTERNAL_OBJECTS_H


  static FT_Error
  _ft_face_scale_advances( FT_Face    face,
                           FT_Fixed*  advances,
                           FT_UInt    count,
                           FT_Int32   flags )
  {
    FT_Fixed  scale;
    FT_UInt   nn;


    if ( flags & FT_LOAD_NO_SCALE )
      return FT_Err_Ok;

    if ( !face->size )
      return FT_THROW( Invalid_Size_Handle );

    if ( flags & FT_LOAD_VERTICAL_LAYOUT )
      scale = face->size->metrics.y_scale;
    else
      scale = face->size->metrics.x_scale;

    /* this must be the same scaling as to get linear{Hori,Vert}Advance */
    /* (see `FT_Load_Glyph' implementation in src/base/ftobjs.c)        */

    for ( nn = 0; nn < count; nn++ )
      advances[nn] = FT_MulDiv( advances[nn], scale, 64 );

    return FT_Err_Ok;
  }


   /* at the moment, we can perform fast advance retrieval only in */
   /* the following cases:                                         */
   /*                                                              */
   /*  - unscaled load                                             */
   /*  - unhinted load                                             */
   /*  - light-hinted load                                         */
   /*  - if a variations font, it must have an `HVAR' or `VVAR'    */
   /*    table (thus the old MM or GX fonts don't qualify; this    */
   /*    gets checked by the driver-specific functions)            */

#define LOAD_ADVANCE_FAST_CHECK( face, flags )                      \
          ( flags & ( FT_LOAD_NO_SCALE | FT_LOAD_NO_HINTING )    || \
            FT_LOAD_TARGET_MODE( flags ) == FT_RENDER_MODE_LIGHT )


  /* documentation is in ftadvanc.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_Advance( FT_Face    face,
                  FT_UInt    gindex,
                  FT_Int32   flags,
                  FT_Fixed  *padvance )
  {
    FT_Face_GetAdvancesFunc  func;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( !padvance )
      return FT_THROW( Invalid_Argument );

    if ( gindex >= (FT_UInt)face->num_glyphs )
      return FT_THROW( Invalid_Glyph_Index );

    func = face->driver->clazz->get_advances;
    if ( func && LOAD_ADVANCE_FAST_CHECK( face, flags ) )
    {
      FT_Error  error;


      error = func( face, gindex, 1, flags, padvance );
      if ( !error )
        return _ft_face_scale_advances( face, padvance, 1, flags );

      if ( FT_ERR_NEQ( error, Unimplemented_Feature ) )
        return error;
    }

    return FT_Get_Advances( face, gindex, 1, flags, padvance );
  }


  /* documentation is in ftadvanc.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_Advances( FT_Face    face,
                   FT_UInt    start,
                   FT_UInt    count,
                   FT_Int32   flags,
                   FT_Fixed  *padvances )
  {
    FT_Error  error = FT_Err_Ok;

    FT_Face_GetAdvancesFunc  func;

    FT_UInt  num, end, nn;
    FT_Int   factor;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( !padvances )
      return FT_THROW( Invalid_Argument );

    num = (FT_UInt)face->num_glyphs;
    end = start + count;
    if ( start >= num || end < start || end > num )
      return FT_THROW( Invalid_Glyph_Index );

    if ( count == 0 )
      return FT_Err_Ok;

    func = face->driver->clazz->get_advances;
    if ( func && LOAD_ADVANCE_FAST_CHECK( face, flags ) )
    {
      error = func( face, start, count, flags, padvances );
      if ( !error )
        return _ft_face_scale_advances( face, padvances, count, flags );

      if ( FT_ERR_NEQ( error, Unimplemented_Feature ) )
        return error;
    }

    error = FT_Err_Ok;

    if ( flags & FT_ADVANCE_FLAG_FAST_ONLY )
      return FT_THROW( Unimplemented_Feature );

    flags |= (FT_UInt32)FT_LOAD_ADVANCE_ONLY;
    factor = ( flags & FT_LOAD_NO_SCALE ) ? 1 : 1024;
    for ( nn = 0; nn < count; nn++ )
    {
      error = FT_Load_Glyph( face, start + nn, flags );
      if ( error )
        break;

      /* scale from 26.6 to 16.16, unless NO_SCALE was requested */
      padvances[nn] = ( flags & FT_LOAD_VERTICAL_LAYOUT )
                      ? face->glyph->advance.y * factor
                      : face->glyph->advance.x * factor;
    }

    return error;
  }


/* END */
