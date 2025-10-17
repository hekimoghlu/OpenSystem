/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
#include FT_INTERNAL_SFNT_H
#include FT_INTERNAL_TRUETYPE_TYPES_H
#include FT_COLOR_H


#ifdef TT_CONFIG_OPTION_COLOR_LAYERS

  static
  const FT_Palette_Data  null_palette_data = { 0, NULL, NULL, 0, NULL };


  /* documentation is in ftcolor.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Palette_Data_Get( FT_Face           face,
                       FT_Palette_Data  *apalette_data )
  {
    if ( !face )
      return FT_THROW( Invalid_Face_Handle );
    if ( !apalette_data)
      return FT_THROW( Invalid_Argument );

    if ( FT_IS_SFNT( face ) )
      *apalette_data = ( (TT_Face)face )->palette_data;
    else
      *apalette_data = null_palette_data;

    return FT_Err_Ok;
  }


  /* documentation is in ftcolor.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Palette_Select( FT_Face     face,
                     FT_UShort   palette_index,
                     FT_Color*  *apalette )
  {
    FT_Error  error;

    TT_Face       ttface;
    SFNT_Service  sfnt;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( !FT_IS_SFNT( face ) )
    {
      if ( apalette )
        *apalette = NULL;

      return FT_Err_Ok;
    }

    ttface = (TT_Face)face;
    sfnt   = (SFNT_Service)ttface->sfnt;

    error = sfnt->set_palette( ttface, palette_index );
    if ( error )
      return error;

    ttface->palette_index = palette_index;

    if ( apalette )
      *apalette = ttface->palette;

    return FT_Err_Ok;
  }


  /* documentation is in ftcolor.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Palette_Set_Foreground_Color( FT_Face   face,
                                   FT_Color  foreground_color )
  {
    TT_Face  ttface;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( !FT_IS_SFNT( face ) )
      return FT_Err_Ok;

    ttface = (TT_Face)face;

    ttface->foreground_color      = foreground_color;
    ttface->have_foreground_color = 1;

    return FT_Err_Ok;
  }

#else /* !TT_CONFIG_OPTION_COLOR_LAYERS */

  FT_EXPORT_DEF( FT_Error )
  FT_Palette_Data_Get( FT_Face           face,
                       FT_Palette_Data  *apalette_data )
  {
    FT_UNUSED( face );
    FT_UNUSED( apalette_data );


    return FT_THROW( Unimplemented_Feature );
  }


  FT_EXPORT_DEF( FT_Error )
  FT_Palette_Select( FT_Face     face,
                     FT_UShort   palette_index,
                     FT_Color*  *apalette )
  {
    FT_UNUSED( face );
    FT_UNUSED( palette_index );
    FT_UNUSED( apalette );


    return FT_THROW( Unimplemented_Feature );
  }


  FT_EXPORT_DEF( FT_Error )
  FT_Palette_Set_Foreground_Color( FT_Face   face,
                                   FT_Color  foreground_color )
  {
    FT_UNUSED( face );
    FT_UNUSED( foreground_color );


    return FT_THROW( Unimplemented_Feature );
  }

#endif /* !TT_CONFIG_OPTION_COLOR_LAYERS */


/* END */
