/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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
/*
   * A `CF2_Blues' object stores the blue zones (horizontal alignment
   * zones) of a font.  These are specified in the CFF private dictionary
   * by `BlueValues', `OtherBlues', `FamilyBlues', and `FamilyOtherBlues'.
   * Each zone is defined by a top and bottom edge in character space.
   * Further, each zone is either a top zone or a bottom zone, as recorded
   * by `bottomZone'.
   *
   * The maximum number of `BlueValues' and `FamilyBlues' is 7 each.
   * However, these are combined to produce a total of 7 zones.
   * Similarly, the maximum number of `OtherBlues' and `FamilyOtherBlues'
   * is 5 and these are combined to produce an additional 5 zones.
   *
   * Blue zones are used to `capture' hints and force them to a common
   * alignment point.  This alignment is recorded in device space in
   * `dsFlatEdge'.  Except for this value, a `CF2_Blues' object could be
   * constructed independently of scaling.  Construction may occur once
   * the matrix is known.  Other features implemented in the Capture
   * method are overshoot suppression, overshoot enforcement, and Blue
   * Boost.
   *
   * Capture is determined by `BlueValues' and `OtherBlues', but the
   * alignment point may be adjusted to the scaled flat edge of
   * `FamilyBlues' or `FamilyOtherBlues'.  No alignment is done to the
   * curved edge of a zone.
   *
   */


#ifndef PSBLUES_H_
#define PSBLUES_H_


#include "psglue.h"


FT_BEGIN_HEADER


  /*
   * `CF2_Hint' is shared by `cf2hints.h' and
   * `cf2blues.h', but `cf2blues.h' depends on
   * `cf2hints.h', so define it here.  Note: The typedef is in
   * `cf2glue.h'.
   *
   */
  enum
  {
    CF2_GhostBottom = 0x1,  /* a single bottom edge           */
    CF2_GhostTop    = 0x2,  /* a single top edge              */
    CF2_PairBottom  = 0x4,  /* the bottom edge of a stem hint */
    CF2_PairTop     = 0x8,  /* the top edge of a stem hint    */
    CF2_Locked      = 0x10, /* this edge has been aligned     */
                            /* by a blue zone                 */
    CF2_Synthetic   = 0x20  /* this edge was synthesized      */
  };


  /*
   * Default value for OS/2 typoAscender/Descender when their difference
   * is not equal to `unitsPerEm'.  The default is based on -250 and 1100
   * in `CF2_Blues', assuming 1000 units per em here.
   *
   */
  enum
  {
    CF2_ICF_Top    = cf2_intToFixed(  880 ),
    CF2_ICF_Bottom = cf2_intToFixed( -120 )
  };


  /*
   * Constant used for hint adjustment and for synthetic em box hint
   * placement.
   */
#define CF2_MIN_COUNTER  cf2_doubleToFixed( 0.5 )


  /* shared typedef is in cf2glue.h */
  struct  CF2_HintRec_
  {
    CF2_UInt  flags;  /* attributes of the edge            */
    size_t    index;  /* index in original stem hint array */
                      /* (if not synthetic)                */
    CF2_Fixed  csCoord;
    CF2_Fixed  dsCoord;
    CF2_Fixed  scale;
  };


  typedef struct  CF2_BlueRec_
  {
    CF2_Fixed  csBottomEdge;
    CF2_Fixed  csTopEdge;
    CF2_Fixed  csFlatEdge; /* may be from either local or Family zones */
    CF2_Fixed  dsFlatEdge; /* top edge of bottom zone or bottom edge   */
                           /* of top zone (rounded)                    */
    FT_Bool  bottomZone;

  } CF2_BlueRec;


  /* max total blue zones is 12 */
  enum
  {
    CF2_MAX_BLUES      = 7,
    CF2_MAX_OTHERBLUES = 5
  };


  typedef struct  CF2_BluesRec_
  {
    CF2_Fixed  scale;
    CF2_UInt   count;
    FT_Bool    suppressOvershoot;
    FT_Bool    doEmBoxHints;

    CF2_Fixed  blueScale;
    CF2_Fixed  blueShift;
    CF2_Fixed  blueFuzz;

    CF2_Fixed  boost;

    CF2_HintRec  emBoxTopEdge;
    CF2_HintRec  emBoxBottomEdge;

    CF2_BlueRec  zone[CF2_MAX_BLUES + CF2_MAX_OTHERBLUES];

  } CF2_BluesRec, *CF2_Blues;


  FT_LOCAL( void )
  cf2_blues_init( CF2_Blues  blues,
                  CF2_Font   font );
  FT_LOCAL( FT_Bool )
  cf2_blues_capture( const CF2_Blues  blues,
                     CF2_Hint         bottomHintEdge,
                     CF2_Hint         topHintEdge );


FT_END_HEADER


#endif /* PSBLUES_H_ */


/* END */
