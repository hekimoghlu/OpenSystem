/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
#ifndef PSHINT_H_
#define PSHINT_H_

FT_BEGIN_HEADER


  enum
  {
    CF2_MAX_HINTS = 96    /* maximum # of hints */
  };


  /*
   * A HintMask object stores a bit mask that specifies which hints in the
   * charstring are active at a given time.  Hints in CFF must be declared
   * at the start, before any drawing operators, with horizontal hints
   * preceding vertical hints.  The HintMask is ordered the same way, with
   * horizontal hints immediately followed by vertical hints.  Clients are
   * responsible for knowing how many of each type are present.
   *
   * The maximum total number of hints is 96, as specified by the CFF
   * specification.
   *
   * A HintMask is built 0 or more times while interpreting a charstring, by
   * the HintMask operator.  There is only one HintMask, but it is built or
   * rebuilt each time there is a hint substitution (HintMask operator) in
   * the charstring.  A default HintMask with all bits set is built if there
   * has been no HintMask operator prior to the first drawing operator.
   *
   */

  typedef struct  CF2_HintMaskRec_
  {
    FT_Error*  error;

    FT_Bool  isValid;
    FT_Bool  isNew;

    size_t  bitCount;
    size_t  byteCount;

    FT_Byte  mask[( CF2_MAX_HINTS + 7 ) / 8];

  } CF2_HintMaskRec, *CF2_HintMask;


  typedef struct  CF2_StemHintRec_
  {
    FT_Bool  used;     /* DS positions are valid         */

    CF2_Fixed  min;    /* original character space value */
    CF2_Fixed  max;

    CF2_Fixed  minDS;  /* DS position after first use    */
    CF2_Fixed  maxDS;

  } CF2_StemHintRec, *CF2_StemHint;


  /*
   * A HintMap object stores a piecewise linear function for mapping
   * y-coordinates from character space to device space, providing
   * appropriate pixel alignment to stem edges.
   *
   * The map is implemented as an array of `CF2_Hint' elements, each
   * representing an edge.  When edges are paired, as from stem hints, the
   * bottom edge must immediately precede the top edge in the array.
   * Element character space AND device space positions must both increase
   * monotonically in the array.  `CF2_Hint' elements are also used as
   * parameters to `cf2_blues_capture'.
   *
   * The `cf2_hintmap_build' method must be called before any drawing
   * operation (beginning with a Move operator) and at each hint
   * substitution (HintMask operator).
   *
   * The `cf2_hintmap_map' method is called to transform y-coordinates at
   * each drawing operation (move, line, curve).
   *
   */

  /* TODO: make this a CF2_ArrStack and add a deep copy method */
  enum
  {
    CF2_MAX_HINT_EDGES = CF2_MAX_HINTS * 2
  };


  typedef struct  CF2_HintMapRec_
  {
    CF2_Font  font;

    /* initial map based on blue zones */
    struct CF2_HintMapRec_*  initialHintMap;

    /* working storage for 2nd pass adjustHints */
    CF2_ArrStack  hintMoves;

    FT_Bool  isValid;
    FT_Bool  hinted;

    CF2_Fixed  scale;
    CF2_UInt   count;

    /* start search from this index */
    CF2_UInt  lastIndex;

    CF2_HintRec  edge[CF2_MAX_HINT_EDGES]; /* 192 */

  } CF2_HintMapRec, *CF2_HintMap;


  FT_LOCAL( FT_Bool )
  cf2_hint_isValid( const CF2_Hint  hint );
  FT_LOCAL( FT_Bool )
  cf2_hint_isTop( const CF2_Hint  hint );
  FT_LOCAL( FT_Bool )
  cf2_hint_isBottom( const CF2_Hint  hint );
  FT_LOCAL( void )
  cf2_hint_lock( CF2_Hint  hint );


  FT_LOCAL( void )
  cf2_hintmap_init( CF2_HintMap   hintmap,
                    CF2_Font      font,
                    CF2_HintMap   initialMap,
                    CF2_ArrStack  hintMoves,
                    CF2_Fixed     scale );
  FT_LOCAL( void )
  cf2_hintmap_build( CF2_HintMap   hintmap,
                     CF2_ArrStack  hStemHintArray,
                     CF2_ArrStack  vStemHintArray,
                     CF2_HintMask  hintMask,
                     CF2_Fixed     hintOrigin,
                     FT_Bool       initialMap );


  /*
   * GlyphPath is a wrapper for drawing operations that scales the
   * coordinates according to the render matrix and HintMap.  It also tracks
   * open paths to control ClosePath and to insert MoveTo for broken fonts.
   *
   */
  typedef struct  CF2_GlyphPathRec_
  {
    /* TODO: gather some of these into a hinting context */

    CF2_Font              font;           /* font instance    */
    CF2_OutlineCallbacks  callbacks;      /* outline consumer */


    CF2_HintMapRec  hintMap;        /* current hint map            */
    CF2_HintMapRec  firstHintMap;   /* saved copy                  */
    CF2_HintMapRec  initialHintMap; /* based on all captured hints */

    CF2_ArrStackRec  hintMoves;  /* list of hint moves for 2nd pass */

    CF2_Fixed  scaleX;         /* matrix a */
    CF2_Fixed  scaleC;         /* matrix c */
    CF2_Fixed  scaleY;         /* matrix d */

    FT_Vector  fractionalTranslation;  /* including deviceXScale */
#if 0
    CF2_Fixed  hShift;    /* character space horizontal shift */
                          /* (for fauxing)                    */
#endif

    FT_Bool  pathIsOpen;     /* true after MoveTo                     */
    FT_Bool  pathIsClosing;  /* true when synthesizing closepath line */
    FT_Bool  darken;         /* true if stem darkening                */
    FT_Bool  moveIsPending;  /* true between MoveTo and offset MoveTo */

    /* references used to call `cf2_hintmap_build', if necessary */
    CF2_ArrStack         hStemHintArray;
    CF2_ArrStack         vStemHintArray;
    CF2_HintMask         hintMask;     /* ptr to the current mask */
    CF2_Fixed            hintOriginY;  /* copy of current origin  */
    const CF2_BluesRec*  blues;

    CF2_Fixed  xOffset;        /* character space offsets */
    CF2_Fixed  yOffset;

    /* character space miter limit threshold */
    CF2_Fixed  miterLimit;
    /* vertical/horizontal snap distance in character space */
    CF2_Fixed  snapThreshold;

    FT_Vector  offsetStart0;  /* first and second points of first */
    FT_Vector  offsetStart1;  /* element with offset applied      */

    /* current point, character space, before offset */
    FT_Vector  currentCS;
    /* current point, device space */
    FT_Vector  currentDS;
    /* start point of subpath, character space */
    FT_Vector  start;

    /* the following members constitute the `queue' of one element */
    FT_Bool  elemIsQueued;
    CF2_Int  prevElemOp;

    FT_Vector  prevElemP0;
    FT_Vector  prevElemP1;
    FT_Vector  prevElemP2;
    FT_Vector  prevElemP3;

  } CF2_GlyphPathRec, *CF2_GlyphPath;


  FT_LOCAL( void )
  cf2_glyphpath_init( CF2_GlyphPath         glyphpath,
                      CF2_Font              font,
                      CF2_OutlineCallbacks  callbacks,
                      CF2_Fixed             scaleY,
                      /* CF2_Fixed hShift, */
                      CF2_ArrStack          hStemHintArray,
                      CF2_ArrStack          vStemHintArray,
                      CF2_HintMask          hintMask,
                      CF2_Fixed             hintOrigin,
                      const CF2_Blues       blues,
                      const FT_Vector*      fractionalTranslation );
  FT_LOCAL( void )
  cf2_glyphpath_finalize( CF2_GlyphPath  glyphpath );

  FT_LOCAL( void )
  cf2_glyphpath_moveTo( CF2_GlyphPath  glyphpath,
                        CF2_Fixed      x,
                        CF2_Fixed      y );
  FT_LOCAL( void )
  cf2_glyphpath_lineTo( CF2_GlyphPath  glyphpath,
                        CF2_Fixed      x,
                        CF2_Fixed      y );
  FT_LOCAL( void )
  cf2_glyphpath_curveTo( CF2_GlyphPath  glyphpath,
                         CF2_Fixed      x1,
                         CF2_Fixed      y1,
                         CF2_Fixed      x2,
                         CF2_Fixed      y2,
                         CF2_Fixed      x3,
                         CF2_Fixed      y3 );
  FT_LOCAL( void )
  cf2_glyphpath_closeOpenPath( CF2_GlyphPath  glyphpath );


FT_END_HEADER


#endif /* PSHINT_H_ */


/* END */
