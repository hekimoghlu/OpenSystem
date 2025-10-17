/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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

//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Check whether variables fit within packing limits according to the packing rules from the GLSL ES
// 1.00.17 spec, Appendix A, section 7.

#ifndef COMPILER_TRANSLATOR_VARIABLEPACKER_H_
#define COMPILER_TRANSLATOR_VARIABLEPACKER_H_

#include <vector>

#include <GLSLANG/ShaderLang.h>

namespace sh
{

// Gets how many components in a row a data type takes.
int GetTypePackingComponentsPerRow(sh::GLenum type);

// Gets how many rows a data type takes.
int GetTypePackingRows(sh::GLenum type);

// Returns true if the passed in variables pack in maxVectors.
// T should be ShaderVariable or one of the subclasses of ShaderVariable.
bool CheckVariablesInPackingLimits(unsigned int maxVectors,
                                   const std::vector<ShaderVariable> &variables);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_VARIABLEPACKER_H_
