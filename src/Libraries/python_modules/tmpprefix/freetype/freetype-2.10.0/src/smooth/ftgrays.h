/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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
#ifndef FTGRAYS_H_
#define FTGRAYS_H_

#ifdef __cplusplus
  extern "C" {
#endif


#ifdef STANDALONE_
#include "ftimage.h"
#else
#include <ft2build.h>
#include FT_IMAGE_H
#endif


  /**************************************************************************
   *
   * To make ftgrays.h independent from configuration files we check
   * whether FT_EXPORT_VAR has been defined already.
   *
   * On some systems and compilers (Win32 mostly), an extra keyword is
   * necessary to compile the library as a DLL.
   */
#ifndef FT_EXPORT_VAR
#define FT_EXPORT_VAR( x )  extern  x
#endif

  FT_EXPORT_VAR( const FT_Raster_Funcs )  ft_grays_raster;


#ifdef __cplusplus
  }
#endif

#endif /* FTGRAYS_H_ */


/* END */
