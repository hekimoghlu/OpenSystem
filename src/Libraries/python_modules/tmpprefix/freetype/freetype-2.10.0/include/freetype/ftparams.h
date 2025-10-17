/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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
#ifndef FTPARAMS_H_
#define FTPARAMS_H_

#include <ft2build.h>
#include FT_FREETYPE_H

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @section:
   *   parameter_tags
   *
   * @title:
   *   Parameter Tags
   *
   * @abstract:
   *   Macros for driver property and font loading parameter tags.
   *
   * @description:
   *   This section contains macros for the @FT_Parameter structure that are
   *   used with various functions to activate some special functionality or
   *   different behaviour of various components of FreeType.
   *
   */


  /**************************************************************************
   *
   * @enum:
   *   FT_PARAM_TAG_IGNORE_TYPOGRAPHIC_FAMILY
   *
   * @description:
   *   A tag for @FT_Parameter to make @FT_Open_Face ignore typographic
   *   family names in the 'name' table (introduced in OpenType version 1.4).
   *   Use this for backward compatibility with legacy systems that have a
   *   four-faces-per-family restriction.
   *
   * @since:
   *   2.8
   *
   */
#define FT_PARAM_TAG_IGNORE_TYPOGRAPHIC_FAMILY \
          FT_MAKE_TAG( 'i', 'g', 'p', 'f' )


  /* this constant is deprecated */
#define FT_PARAM_TAG_IGNORE_PREFERRED_FAMILY \
          FT_PARAM_TAG_IGNORE_TYPOGRAPHIC_FAMILY


  /**************************************************************************
   *
   * @enum:
   *   FT_PARAM_TAG_IGNORE_TYPOGRAPHIC_SUBFAMILY
   *
   * @description:
   *   A tag for @FT_Parameter to make @FT_Open_Face ignore typographic
   *   subfamily names in the 'name' table (introduced in OpenType version
   *   1.4).  Use this for backward compatibility with legacy systems that
   *   have a four-faces-per-family restriction.
   *
   * @since:
   *   2.8
   *
   */
#define FT_PARAM_TAG_IGNORE_TYPOGRAPHIC_SUBFAMILY \
          FT_MAKE_TAG( 'i', 'g', 'p', 's' )


  /* this constant is deprecated */
#define FT_PARAM_TAG_IGNORE_PREFERRED_SUBFAMILY \
          FT_PARAM_TAG_IGNORE_TYPOGRAPHIC_SUBFAMILY


  /**************************************************************************
   *
   * @enum:
   *   FT_PARAM_TAG_INCREMENTAL
   *
   * @description:
   *   An @FT_Parameter tag to be used with @FT_Open_Face to indicate
   *   incremental glyph loading.
   *
   */
#define FT_PARAM_TAG_INCREMENTAL \
          FT_MAKE_TAG( 'i', 'n', 'c', 'r' )


  /**************************************************************************
   *
   * @enum:
   *   FT_PARAM_TAG_LCD_FILTER_WEIGHTS
   *
   * @description:
   *   An @FT_Parameter tag to be used with @FT_Face_Properties.  The
   *   corresponding argument specifies the five LCD filter weights for a
   *   given face (if using @FT_LOAD_TARGET_LCD, for example), overriding the
   *   global default values or the values set up with
   *   @FT_Library_SetLcdFilterWeights.
   *
   * @since:
   *   2.8
   *
   */
#define FT_PARAM_TAG_LCD_FILTER_WEIGHTS \
          FT_MAKE_TAG( 'l', 'c', 'd', 'f' )


  /**************************************************************************
   *
   * @enum:
   *   FT_PARAM_TAG_RANDOM_SEED
   *
   * @description:
   *   An @FT_Parameter tag to be used with @FT_Face_Properties.  The
   *   corresponding 32bit signed integer argument overrides the font
   *   driver's random seed value with a face-specific one; see @random-seed.
   *
   * @since:
   *   2.8
   *
   */
#define FT_PARAM_TAG_RANDOM_SEED \
          FT_MAKE_TAG( 's', 'e', 'e', 'd' )


  /**************************************************************************
   *
   * @enum:
   *   FT_PARAM_TAG_STEM_DARKENING
   *
   * @description:
   *   An @FT_Parameter tag to be used with @FT_Face_Properties.  The
   *   corresponding Boolean argument specifies whether to apply stem
   *   darkening, overriding the global default values or the values set up
   *   with @FT_Property_Set (see @no-stem-darkening).
   *
   *   This is a passive setting that only takes effect if the font driver or
   *   autohinter honors it, which the CFF, Type~1, and CID drivers always
   *   do, but the autohinter only in 'light' hinting mode (as of version
   *   2.9).
   *
   * @since:
   *   2.8
   *
   */
#define FT_PARAM_TAG_STEM_DARKENING \
          FT_MAKE_TAG( 'd', 'a', 'r', 'k' )


  /**************************************************************************
   *
   * @enum:
   *   FT_PARAM_TAG_UNPATENTED_HINTING
   *
   * @description:
   *   Deprecated, no effect.
   *
   *   Previously: A constant used as the tag of an @FT_Parameter structure
   *   to indicate that unpatented methods only should be used by the
   *   TrueType bytecode interpreter for a typeface opened by @FT_Open_Face.
   *
   */
#define FT_PARAM_TAG_UNPATENTED_HINTING \
          FT_MAKE_TAG( 'u', 'n', 'p', 'a' )


  /* */


FT_END_HEADER


#endif /* FTPARAMS_H_ */


/* END */
