/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 23, 2022.
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
#ifndef FTRASTER_H_
#define FTRASTER_H_


#include <ft2build.h>
#include FT_CONFIG_CONFIG_H
#include FT_IMAGE_H


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * Uncomment the following line if you are using ftraster.c as a
   * standalone module, fully independent of FreeType.
   */
/* #define STANDALONE_ */

  FT_EXPORT_VAR( const FT_Raster_Funcs )  ft_standard_raster;


FT_END_HEADER

#endif /* FTRASTER_H_ */


/* END */
