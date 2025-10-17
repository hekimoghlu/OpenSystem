/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
#include "cffcmap.h"
#include "cffload.h"

#include "cfferrs.h"


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****           CFF STANDARD (AND EXPERT) ENCODING CMAPS            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_CALLBACK_DEF( FT_Error )
  cff_cmap_encoding_init( CFF_CMapStd  cmap,
                          FT_Pointer   pointer )
  {
    TT_Face       face     = (TT_Face)FT_CMAP_FACE( cmap );
    CFF_Font      cff      = (CFF_Font)face->extra.data;
    CFF_Encoding  encoding = &cff->encoding;

    FT_UNUSED( pointer );


    cmap->gids  = encoding->codes;

    return 0;
  }


  FT_CALLBACK_DEF( void )
  cff_cmap_encoding_done( CFF_CMapStd  cmap )
  {
    cmap->gids  = NULL;
  }


  FT_CALLBACK_DEF( FT_UInt )
  cff_cmap_encoding_char_index( CFF_CMapStd  cmap,
                                FT_UInt32    char_code )
  {
    FT_UInt  result = 0;


    if ( char_code < 256 )
      result = cmap->gids[char_code];

    return result;
  }


  FT_CALLBACK_DEF( FT_UInt32 )
  cff_cmap_encoding_char_next( CFF_CMapStd   cmap,
                               FT_UInt32    *pchar_code )
  {
    FT_UInt    result    = 0;
    FT_UInt32  char_code = *pchar_code;


    *pchar_code = 0;

    if ( char_code < 255 )
    {
      FT_UInt  code = (FT_UInt)(char_code + 1);


      for (;;)
      {
        if ( code >= 256 )
          break;

        result = cmap->gids[code];
        if ( result != 0 )
        {
          *pchar_code = code;
          break;
        }

        code++;
      }
    }
    return result;
  }


  FT_DEFINE_CMAP_CLASS(
    cff_cmap_encoding_class_rec,

    sizeof ( CFF_CMapStdRec ),

    (FT_CMap_InitFunc)     cff_cmap_encoding_init,        /* init       */
    (FT_CMap_DoneFunc)     cff_cmap_encoding_done,        /* done       */
    (FT_CMap_CharIndexFunc)cff_cmap_encoding_char_index,  /* char_index */
    (FT_CMap_CharNextFunc) cff_cmap_encoding_char_next,   /* char_next  */

    (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
    (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
    (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
    (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
    (FT_CMap_VariantCharListFunc) NULL   /* variantchar_list */
  )


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****              CFF SYNTHETIC UNICODE ENCODING CMAP              *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_CALLBACK_DEF( const char* )
  cff_sid_to_glyph_name( TT_Face  face,
                         FT_UInt  idx )
  {
    CFF_Font     cff     = (CFF_Font)face->extra.data;
    CFF_Charset  charset = &cff->charset;
    FT_UInt      sid     = charset->sids[idx];


    return cff_index_get_sid_string( cff, sid );
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_cmap_unicode_init( PS_Unicodes  unicodes,
                         FT_Pointer   pointer )
  {
    TT_Face             face    = (TT_Face)FT_CMAP_FACE( unicodes );
    FT_Memory           memory  = FT_FACE_MEMORY( face );
    CFF_Font            cff     = (CFF_Font)face->extra.data;
    CFF_Charset         charset = &cff->charset;
    FT_Service_PsCMaps  psnames = (FT_Service_PsCMaps)cff->psnames;

    FT_UNUSED( pointer );


    /* can't build Unicode map for CID-keyed font */
    /* because we don't know glyph names.         */
    if ( !charset->sids )
      return FT_THROW( No_Unicode_Glyph_Name );

    if ( !psnames->unicodes_init )
      return FT_THROW( Unimplemented_Feature );

    return psnames->unicodes_init( memory,
                                   unicodes,
                                   cff->num_glyphs,
                                   (PS_GetGlyphNameFunc)&cff_sid_to_glyph_name,
                                   (PS_FreeGlyphNameFunc)NULL,
                                   (FT_Pointer)face );
  }


  FT_CALLBACK_DEF( void )
  cff_cmap_unicode_done( PS_Unicodes  unicodes )
  {
    FT_Face    face   = FT_CMAP_FACE( unicodes );
    FT_Memory  memory = FT_FACE_MEMORY( face );


    FT_FREE( unicodes->maps );
    unicodes->num_maps = 0;
  }


  FT_CALLBACK_DEF( FT_UInt )
  cff_cmap_unicode_char_index( PS_Unicodes  unicodes,
                               FT_UInt32    char_code )
  {
    TT_Face             face    = (TT_Face)FT_CMAP_FACE( unicodes );
    CFF_Font            cff     = (CFF_Font)face->extra.data;
    FT_Service_PsCMaps  psnames = (FT_Service_PsCMaps)cff->psnames;


    return psnames->unicodes_char_index( unicodes, char_code );
  }


  FT_CALLBACK_DEF( FT_UInt32 )
  cff_cmap_unicode_char_next( PS_Unicodes  unicodes,
                              FT_UInt32   *pchar_code )
  {
    TT_Face             face    = (TT_Face)FT_CMAP_FACE( unicodes );
    CFF_Font            cff     = (CFF_Font)face->extra.data;
    FT_Service_PsCMaps  psnames = (FT_Service_PsCMaps)cff->psnames;


    return psnames->unicodes_char_next( unicodes, pchar_code );
  }


  FT_DEFINE_CMAP_CLASS(
    cff_cmap_unicode_class_rec,

    sizeof ( PS_UnicodesRec ),

    (FT_CMap_InitFunc)     cff_cmap_unicode_init,        /* init       */
    (FT_CMap_DoneFunc)     cff_cmap_unicode_done,        /* done       */
    (FT_CMap_CharIndexFunc)cff_cmap_unicode_char_index,  /* char_index */
    (FT_CMap_CharNextFunc) cff_cmap_unicode_char_next,   /* char_next  */

    (FT_CMap_CharVarIndexFunc)    NULL,  /* char_var_index   */
    (FT_CMap_CharVarIsDefaultFunc)NULL,  /* char_var_default */
    (FT_CMap_VariantListFunc)     NULL,  /* variant_list     */
    (FT_CMap_CharVariantListFunc) NULL,  /* charvariant_list */
    (FT_CMap_VariantCharListFunc) NULL   /* variantchar_list */
  )


/* END */
