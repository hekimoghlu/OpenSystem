/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
/**************************************************************************
   *
   * This file is automatically included by `ft2build.h`.  Do not include it
   * manually!
   *
   */


#define FT_INTERNAL_OBJECTS_H             <freetype/internal/ftobjs.h>
#define FT_INTERNAL_STREAM_H              <freetype/internal/ftstream.h>
#define FT_INTERNAL_MEMORY_H              <freetype/internal/ftmemory.h>
#define FT_INTERNAL_DEBUG_H               <freetype/internal/ftdebug.h>
#define FT_INTERNAL_CALC_H                <freetype/internal/ftcalc.h>
#define FT_INTERNAL_HASH_H                <freetype/internal/fthash.h>
#define FT_INTERNAL_DRIVER_H              <freetype/internal/ftdrv.h>
#define FT_INTERNAL_TRACE_H               <freetype/internal/fttrace.h>
#define FT_INTERNAL_GLYPH_LOADER_H        <freetype/internal/ftgloadr.h>
#define FT_INTERNAL_SFNT_H                <freetype/internal/sfnt.h>
#define FT_INTERNAL_SERVICE_H             <freetype/internal/ftserv.h>
#define FT_INTERNAL_RFORK_H               <freetype/internal/ftrfork.h>
#define FT_INTERNAL_VALIDATE_H            <freetype/internal/ftvalid.h>

#define FT_INTERNAL_TRUETYPE_TYPES_H      <freetype/internal/tttypes.h>
#define FT_INTERNAL_TYPE1_TYPES_H         <freetype/internal/t1types.h>

#define FT_INTERNAL_POSTSCRIPT_AUX_H      <freetype/internal/psaux.h>
#define FT_INTERNAL_POSTSCRIPT_HINTS_H    <freetype/internal/pshints.h>
#define FT_INTERNAL_POSTSCRIPT_PROPS_H    <freetype/internal/ftpsprop.h>

#define FT_INTERNAL_AUTOHINT_H            <freetype/internal/autohint.h>

#define FT_INTERNAL_CFF_TYPES_H           <freetype/internal/cfftypes.h>
#define FT_INTERNAL_CFF_OBJECTS_TYPES_H   <freetype/internal/cffotypes.h>


#if defined( _MSC_VER )      /* Visual C++ (and Intel C++) */

  /* We disable the warning `conditional expression is constant' here */
  /* in order to compile cleanly with the maximum level of warnings.  */
  /* In particular, the warning complains about stuff like `while(0)' */
  /* which is very useful in macro definitions.  There is no benefit  */
  /* in having it enabled.                                            */
#pragma warning( disable : 4127 )

#endif /* _MSC_VER */


/* END */
