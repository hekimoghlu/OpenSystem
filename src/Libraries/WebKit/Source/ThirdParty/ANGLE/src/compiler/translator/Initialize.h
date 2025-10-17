/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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

#ifndef COMPILER_TRANSLATOR_INITIALIZE_H_
#define COMPILER_TRANSLATOR_INITIALIZE_H_

#include "compiler/translator/Common.h"
#include "compiler/translator/Compiler.h"
#include "compiler/translator/SymbolTable.h"

namespace sh
{

void InitExtensionBehavior(const ShBuiltInResources &resources,
                           TExtensionBehavior &extensionBehavior);

// Resets the behavior of the extensions listed in |extensionBehavior| to the
// undefined state. These extensions will only be those initially supported in
// the ShBuiltInResources object for this compiler instance. All other
// extensions will remain unsupported.
void ResetExtensionBehavior(const ShBuiltInResources &resources,
                            TExtensionBehavior &extensionBehavior,
                            const ShCompileOptions &compileOptions);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_INITIALIZE_H_
