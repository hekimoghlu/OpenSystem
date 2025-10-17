/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
/* This header file can be included multiple times. */
  /* Define `COVERAGE' as needed.                     */


  /* Add new coverages here.  The first and second arguments are the   */
  /* coverage name in lowercase and uppercase, respectively, followed  */
  /* by a description string.  The last four arguments are the four    */
  /* characters defining the corresponding OpenType feature.           */

#if 0
  /* XXX: It's not possible to define blue zone characters in advance. */
  COVERAGE( alternative_fractions, ALTERNATIVE_FRACTIONS,
            "alternative fractions",
            'a', 'f', 'r', 'c' )
#endif

  COVERAGE( petite_capitals_from_capitals, PETITE_CAPITALS_FROM_CAPITALS,
            "petite capitals from capitals",
            'c', '2', 'c', 'p' )

  COVERAGE( small_capitals_from_capitals, SMALL_CAPITALS_FROM_CAPITALS,
            "small capitals from capitals",
            'c', '2', 's', 'c' )

#if 0
  /* XXX: Only digits are in this coverage, however, both normal style */
  /*      and oldstyle representation forms are possible.              */
  COVERAGE( denominators, DENOMINATORS,
            "denominators",
            'd', 'n', 'o', 'm' )
#endif

#if 0
  /* XXX: It's not possible to define blue zone characters in advance. */
  COVERAGE( fractions, FRACTIONS,
            "fractions",
            'f', 'r', 'a', 'c' )
#endif

#if 0
  /* XXX: Only digits are in this coverage, however, both normal style */
  /*      and oldstyle representation forms are possible.              */
  COVERAGE( numerators, NUMERATORS,
            "numerators",
            'n', 'u', 'm', 'r' )
#endif

  COVERAGE( ordinals, ORDINALS,
            "ordinals",
            'o', 'r', 'd', 'n' )

  COVERAGE( petite_capitals, PETITE_CAPITALS,
            "petite capitals",
            'p', 'c', 'a', 'p' )

  COVERAGE( ruby, RUBY,
            "ruby",
            'r', 'u', 'b', 'y' )

  COVERAGE( scientific_inferiors, SCIENTIFIC_INFERIORS,
            "scientific inferiors",
            's', 'i', 'n', 'f' )

  COVERAGE( small_capitals, SMALL_CAPITALS,
            "small capitals",
            's', 'm', 'c', 'p' )

  COVERAGE( subscript, SUBSCRIPT,
            "subscript",
            's', 'u', 'b', 's' )

  COVERAGE( superscript, SUPERSCRIPT,
            "superscript",
            's', 'u', 'p', 's' )

  COVERAGE( titling, TITLING,
            "titling",
            't', 'i', 't', 'l' )

#if 0
  /* to be always excluded */
  COVERAGE(nalt, 'n', 'a', 'l', 't'); /* Alternate Annotation Forms (?) */
  COVERAGE(ornm, 'o', 'r', 'n', 'm'); /* Ornaments (?) */
#endif


/* END */
