/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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
#ifndef PSAUXMOD_H_
#define PSAUXMOD_H_


#include <ft2build.h>
#include FT_MODULE_H

#include FT_INTERNAL_POSTSCRIPT_AUX_H


FT_BEGIN_HEADER


  FT_CALLBACK_TABLE
  const CFF_Builder_FuncsRec  cff_builder_funcs;

  FT_CALLBACK_TABLE
  const PS_Builder_FuncsRec   ps_builder_funcs;


  FT_EXPORT_VAR( const FT_Module_Class )  psaux_driver_class;


FT_END_HEADER

#endif /* PSAUXMOD_H_ */


/* END */
