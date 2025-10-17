/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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
#ifndef PSGLUE_H_
#define PSGLUE_H_


/* common includes for other modules */
#include "pserror.h"
#include "psfixed.h"
#include "psarrst.h"
#include "psread.h"


FT_BEGIN_HEADER


  /* rendering parameters */

  /* apply hints to rendered glyphs */
#define CF2_FlagsHinted    1
  /* for testing */
#define CF2_FlagsDarkened  2

  /* type for holding the flags */
  typedef CF2_Int  CF2_RenderingFlags;


  /* elements of a glyph outline */
  typedef enum  CF2_PathOp_
  {
    CF2_PathOpMoveTo = 1,     /* change the current point */
    CF2_PathOpLineTo = 2,     /* line                     */
    CF2_PathOpQuadTo = 3,     /* quadratic curve          */
    CF2_PathOpCubeTo = 4      /* cubic curve              */

  } CF2_PathOp;


  /* a matrix of fixed point values */
  typedef struct  CF2_Matrix_
  {
    CF2_F16Dot16  a;
    CF2_F16Dot16  b;
    CF2_F16Dot16  c;
    CF2_F16Dot16  d;
    CF2_F16Dot16  tx;
    CF2_F16Dot16  ty;

  } CF2_Matrix;


  /* these typedefs are needed by more than one header file */
  /* and gcc compiler doesn't allow redefinition            */
  typedef struct CF2_FontRec_  CF2_FontRec, *CF2_Font;
  typedef struct CF2_HintRec_  CF2_HintRec, *CF2_Hint;


  /* A common structure for all callback parameters.                       */
  /*                                                                       */
  /* Some members may be unused.  For example, `pt0' is not used for       */
  /* `moveTo' and `pt3' is not used for `quadTo'.  The initial point `pt0' */
  /* is included for each path element for generality; curve conversions   */
  /* need it.  The `op' parameter allows one function to handle multiple   */
  /* element types.                                                        */

  typedef struct  CF2_CallbackParamsRec_
  {
    FT_Vector  pt0;
    FT_Vector  pt1;
    FT_Vector  pt2;
    FT_Vector  pt3;

    CF2_Int  op;

  } CF2_CallbackParamsRec, *CF2_CallbackParams;


  /* forward reference */
  typedef struct CF2_OutlineCallbacksRec_  CF2_OutlineCallbacksRec,
                                           *CF2_OutlineCallbacks;

  /* callback function pointers */
  typedef void
  (*CF2_Callback_Type)( CF2_OutlineCallbacks      callbacks,
                        const CF2_CallbackParams  params );


  struct  CF2_OutlineCallbacksRec_
  {
    CF2_Callback_Type  moveTo;
    CF2_Callback_Type  lineTo;
    CF2_Callback_Type  quadTo;
    CF2_Callback_Type  cubeTo;

    CF2_Int  windingMomentum;    /* for winding order detection */

    FT_Memory  memory;
    FT_Error*  error;
  };


FT_END_HEADER


#endif /* PSGLUE_H_ */


/* END */
