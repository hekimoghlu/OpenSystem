/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DriverUniformMetal:
//   Struct defining the default driver uniforms for direct and SpirV based ANGLE translation
//

#ifndef COMPILER_TRANSLATOR_MSL_DRIVERUNIFORMMETAL_H_
#define COMPILER_TRANSLATOR_MSL_DRIVERUNIFORMMETAL_H_

#include "compiler/translator/tree_util/DriverUniform.h"

namespace sh
{

class DriverUniformMetal : public DriverUniformExtended
{
  public:
    DriverUniformMetal(DriverUniformMode mode) : DriverUniformExtended(mode) {}
    DriverUniformMetal() : DriverUniformExtended(DriverUniformMode::InterfaceBlock) {}
    ~DriverUniformMetal() override {}

    TIntermTyped *getCoverageMaskField() const;

  protected:
    TFieldList *createUniformFields(TSymbolTable *symbolTable) override;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_MSL_DRIVERUNIFORMMETAL_H_
