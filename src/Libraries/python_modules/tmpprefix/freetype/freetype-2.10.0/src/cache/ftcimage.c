/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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
#include FT_CACHE_H
#include "ftcimage.h"
#include FT_INTERNAL_MEMORY_H
#include FT_INTERNAL_OBJECTS_H

#include "ftccback.h"
#include "ftcerror.h"


  /* finalize a given glyph image node */
  FT_LOCAL_DEF( void )
  ftc_inode_free( FTC_Node   ftcinode,
                  FTC_Cache  cache )
  {
    FTC_INode  inode = (FTC_INode)ftcinode;
    FT_Memory  memory = cache->memory;


    if ( inode->glyph )
    {
      FT_Done_Glyph( inode->glyph );
      inode->glyph = NULL;
    }

    FTC_GNode_Done( FTC_GNODE( inode ), cache );
    FT_FREE( inode );
  }


  FT_LOCAL_DEF( void )
  FTC_INode_Free( FTC_INode  inode,
                  FTC_Cache  cache )
  {
    ftc_inode_free( FTC_NODE( inode ), cache );
  }


  /* initialize a new glyph image node */
  FT_LOCAL_DEF( FT_Error )
  FTC_INode_New( FTC_INode   *pinode,
                 FTC_GQuery   gquery,
                 FTC_Cache    cache )
  {
    FT_Memory  memory = cache->memory;
    FT_Error   error;
    FTC_INode  inode  = NULL;


    if ( !FT_NEW( inode ) )
    {
      FTC_GNode         gnode  = FTC_GNODE( inode );
      FTC_Family        family = gquery->family;
      FT_UInt           gindex = gquery->gindex;
      FTC_IFamilyClass  clazz  = FTC_CACHE_IFAMILY_CLASS( cache );


      /* initialize its inner fields */
      FTC_GNode_Init( gnode, gindex, family );

      /* we will now load the glyph image */
      error = clazz->family_load_glyph( family, gindex, cache,
                                        &inode->glyph );
      if ( error )
      {
        FTC_INode_Free( inode, cache );
        inode = NULL;
      }
    }

    *pinode = inode;
    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  ftc_inode_new( FTC_Node   *ftcpinode,
                 FT_Pointer  ftcgquery,
                 FTC_Cache   cache )
  {
    FTC_INode  *pinode = (FTC_INode*)ftcpinode;
    FTC_GQuery  gquery = (FTC_GQuery)ftcgquery;


    return FTC_INode_New( pinode, gquery, cache );
  }


  FT_LOCAL_DEF( FT_Offset )
  ftc_inode_weight( FTC_Node   ftcinode,
                    FTC_Cache  ftccache )
  {
    FTC_INode  inode = (FTC_INode)ftcinode;
    FT_Offset  size  = 0;
    FT_Glyph   glyph = inode->glyph;

    FT_UNUSED( ftccache );


    switch ( glyph->format )
    {
    case FT_GLYPH_FORMAT_BITMAP:
      {
        FT_BitmapGlyph  bitg;


        bitg = (FT_BitmapGlyph)glyph;
        size = bitg->bitmap.rows * (FT_Offset)FT_ABS( bitg->bitmap.pitch ) +
               sizeof ( *bitg );
      }
      break;

    case FT_GLYPH_FORMAT_OUTLINE:
      {
        FT_OutlineGlyph  outg;


        outg = (FT_OutlineGlyph)glyph;
        size = (FT_Offset)outg->outline.n_points *
                 ( sizeof ( FT_Vector ) + sizeof ( FT_Byte ) ) +
               (FT_Offset)outg->outline.n_contours * sizeof ( FT_Short ) +
               sizeof ( *outg );
      }
      break;

    default:
      ;
    }

    size += sizeof ( *inode );
    return size;
  }


#if 0

  FT_LOCAL_DEF( FT_Offset )
  FTC_INode_Weight( FTC_INode  inode )
  {
    return ftc_inode_weight( FTC_NODE( inode ), NULL );
  }

#endif /* 0 */


/* END */
