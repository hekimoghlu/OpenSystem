/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 18, 2024.
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
#ifndef PSFONT_H_
#define PSFONT_H_


#include FT_SERVICE_CFF_TABLE_LOAD_H

#include "psft.h"
#include "psblues.h"


FT_BEGIN_HEADER


#define CF2_OPERAND_STACK_SIZE  48
#define CF2_MAX_SUBR            16 /* maximum subroutine nesting;         */
                                   /* only 10 are allowed but there exist */
                                   /* fonts like `HiraKakuProN-W3.ttf'    */
                                   /* (Hiragino Kaku Gothic ProN W3;      */
                                   /* 8.2d6e1; 2014-12-19) that exceed    */
                                   /* this limit                          */
#define CF2_STORAGE_SIZE        32


  /* typedef is in `cf2glue.h' */
  struct  CF2_FontRec_
  {
    FT_Memory  memory;
    FT_Error   error;     /* shared error for this instance */

    FT_Bool             isT1;
    FT_Bool             isCFF2;
    CF2_RenderingFlags  renderingFlags;

    /* variables that depend on Transform:  */
    /* the following have zero translation; */
    /* inner * outer = font * original      */

    CF2_Matrix  currentTransform;  /* original client matrix           */
    CF2_Matrix  innerTransform;    /* for hinting; erect, scaled       */
    CF2_Matrix  outerTransform;    /* post hinting; includes rotations */
    CF2_Fixed   ppem;              /* transform-dependent              */

    /* variation data */
    CFF_BlendRec  blend;            /* cached charstring blend vector  */
    CF2_UInt      vsindex;          /* current vsindex                 */
    CF2_UInt      lenNDV;           /* current length NDV or zero      */
    FT_Fixed*     NDV;              /* ptr to current NDV or NULL      */

    CF2_Int  unitsPerEm;

    CF2_Fixed  syntheticEmboldeningAmountX;   /* character space units */
    CF2_Fixed  syntheticEmboldeningAmountY;   /* character space units */

    /* FreeType related members */
    CF2_OutlineRec  outline;       /* freetype glyph outline functions */
    PS_Decoder*     decoder;
    CFF_SubFont     lastSubfont;              /* FreeType parsed data; */
                                              /* top font or subfont   */

    /* these flags can vary from one call to the next */
    FT_Bool  hinted;
    FT_Bool  darkened;       /* true if stemDarkened or synthetic bold */
                             /* i.e. darkenX != 0 || darkenY != 0      */
    FT_Bool  stemDarkened;

    FT_Int  darkenParams[8];              /* 1000 unit character space */

    /* variables that depend on both FontDict and Transform */
    CF2_Fixed  stdVW;     /* in character space; depends on dict entry */
    CF2_Fixed  stdHW;     /* in character space; depends on dict entry */
    CF2_Fixed  darkenX;                    /* character space units    */
    CF2_Fixed  darkenY;                    /* depends on transform     */
                                           /* and private dict (StdVW) */
    FT_Bool  reverseWinding;               /* darken assuming          */
                                           /* counterclockwise winding */

    CF2_BluesRec  blues;                         /* computed zone data */

    FT_Service_CFFLoad  cffload;           /* pointer to cff functions */
  };


  FT_LOCAL( FT_Error )
  cf2_getGlyphOutline( CF2_Font           font,
                       CF2_Buffer         charstring,
                       const CF2_Matrix*  transform,
                       CF2_F16Dot16*      glyphWidth );


FT_END_HEADER


#endif /* PSFONT_H_ */


/* END */
