/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
#include FT_INTERNAL_OBJECTS_H
#include FT_OUTLINE_H
#include "ftrend1.h"
#include "ftraster.h"

#include "rasterrs.h"


  /* initialize renderer -- init its raster */
  static FT_Error
  ft_raster1_init( FT_Renderer  render )
  {
    render->clazz->raster_class->raster_reset( render->raster, NULL, 0 );

    return FT_Err_Ok;
  }


  /* set render-specific mode */
  static FT_Error
  ft_raster1_set_mode( FT_Renderer  render,
                       FT_ULong     mode_tag,
                       FT_Pointer   data )
  {
    /* we simply pass it to the raster */
    return render->clazz->raster_class->raster_set_mode( render->raster,
                                                         mode_tag,
                                                         data );
  }


  /* transform a given glyph image */
  static FT_Error
  ft_raster1_transform( FT_Renderer       render,
                        FT_GlyphSlot      slot,
                        const FT_Matrix*  matrix,
                        const FT_Vector*  delta )
  {
    FT_Error error = FT_Err_Ok;


    if ( slot->format != render->glyph_format )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( matrix )
      FT_Outline_Transform( &slot->outline, matrix );

    if ( delta )
      FT_Outline_Translate( &slot->outline, delta->x, delta->y );

  Exit:
    return error;
  }


  /* return the glyph's control box */
  static void
  ft_raster1_get_cbox( FT_Renderer   render,
                       FT_GlyphSlot  slot,
                       FT_BBox*      cbox )
  {
    FT_ZERO( cbox );

    if ( slot->format == render->glyph_format )
      FT_Outline_Get_CBox( &slot->outline, cbox );
  }


  /* convert a slot's glyph image into a bitmap */
  static FT_Error
  ft_raster1_render( FT_Renderer       render,
                     FT_GlyphSlot      slot,
                     FT_Render_Mode    mode,
                     const FT_Vector*  origin )
  {
    FT_Error     error   = FT_Err_Ok;
    FT_Outline*  outline = &slot->outline;
    FT_Bitmap*   bitmap  = &slot->bitmap;
    FT_Memory    memory  = render->root.memory;
    FT_Pos       x_shift = 0;
    FT_Pos       y_shift = 0;

    FT_Raster_Params  params;


    /* check glyph image format */
    if ( slot->format != render->glyph_format )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* check rendering mode */
    if ( mode != FT_RENDER_MODE_MONO )
    {
      /* raster1 is only capable of producing monochrome bitmaps */
      return FT_THROW( Cannot_Render_Glyph );
    }

    /* release old bitmap buffer */
    if ( slot->internal->flags & FT_GLYPH_OWN_BITMAP )
    {
      FT_FREE( bitmap->buffer );
      slot->internal->flags &= ~FT_GLYPH_OWN_BITMAP;
    }

    if ( ft_glyphslot_preset_bitmap( slot, mode, origin ) )
    {
      error = FT_THROW( Raster_Overflow );
      goto Exit;
    }

    /* allocate new one */
    if ( FT_ALLOC_MULT( bitmap->buffer, bitmap->rows, bitmap->pitch ) )
      goto Exit;

    slot->internal->flags |= FT_GLYPH_OWN_BITMAP;

    x_shift = -slot->bitmap_left * 64;
    y_shift = ( (FT_Int)bitmap->rows - slot->bitmap_top ) * 64;

    if ( origin )
    {
      x_shift += origin->x;
      y_shift += origin->y;
    }

    /* translate outline to render it into the bitmap */
    if ( x_shift || y_shift )
      FT_Outline_Translate( outline, x_shift, y_shift );

    /* set up parameters */
    params.target = bitmap;
    params.source = outline;
    params.flags  = FT_RASTER_FLAG_DEFAULT;

    /* render outline into the bitmap */
    error = render->raster_render( render->raster, &params );

  Exit:
    if ( !error )
      /* everything is fine; the glyph is now officially a bitmap */
      slot->format = FT_GLYPH_FORMAT_BITMAP;
    else if ( slot->internal->flags & FT_GLYPH_OWN_BITMAP )
    {
      FT_FREE( bitmap->buffer );
      slot->internal->flags &= ~FT_GLYPH_OWN_BITMAP;
    }

    if ( x_shift || y_shift )
      FT_Outline_Translate( outline, -x_shift, -y_shift );

    return error;
  }


  FT_DEFINE_RENDERER(
    ft_raster1_renderer_class,

      FT_MODULE_RENDERER,
      sizeof ( FT_RendererRec ),

      "raster1",
      0x10000L,
      0x20000L,

      NULL,    /* module specific interface */

      (FT_Module_Constructor)ft_raster1_init,  /* module_init   */
      (FT_Module_Destructor) NULL,             /* module_done   */
      (FT_Module_Requester)  NULL,             /* get_interface */

    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Renderer_RenderFunc)   ft_raster1_render,     /* render_glyph    */
    (FT_Renderer_TransformFunc)ft_raster1_transform,  /* transform_glyph */
    (FT_Renderer_GetCBoxFunc)  ft_raster1_get_cbox,   /* get_glyph_cbox  */
    (FT_Renderer_SetModeFunc)  ft_raster1_set_mode,   /* set_mode        */

    (FT_Raster_Funcs*)&ft_standard_raster             /* raster_class    */
  )


/* END */
