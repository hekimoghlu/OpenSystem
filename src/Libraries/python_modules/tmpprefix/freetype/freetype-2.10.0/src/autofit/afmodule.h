/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#ifndef AFMODULE_H_
#define AFMODULE_H_

#include <ft2build.h>
#include FT_INTERNAL_OBJECTS_H
#include FT_MODULE_H


FT_BEGIN_HEADER


  /*
   * This is the `extended' FT_Module structure that holds the
   * autofitter's global data.
   */

  typedef struct  AF_ModuleRec_
  {
    FT_ModuleRec  root;

    FT_UInt       fallback_style;
    FT_UInt       default_script;
#ifdef AF_CONFIG_OPTION_USE_WARPER
    FT_Bool       warping;
#endif
    FT_Bool       no_stem_darkening;
    FT_Int        darken_params[8];

  } AF_ModuleRec, *AF_Module;


FT_DECLARE_MODULE( autofit_module_class )


FT_END_HEADER

#endif /* AFMODULE_H_ */


/* END */
