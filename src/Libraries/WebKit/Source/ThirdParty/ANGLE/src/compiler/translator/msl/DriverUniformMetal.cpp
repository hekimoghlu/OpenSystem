/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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

#include "compiler/translator/msl/DriverUniformMetal.h"
#include "compiler/translator/tree_util/BuiltIn.h"
#include "compiler/translator/tree_util/DriverUniform.h"
#include "compiler/translator/tree_util/IntermNode_util.h"

namespace sh
{

namespace
{

// Metal specific driver uniforms
constexpr const char kXfbBufferOffsets[]       = "xfbBufferOffsets";
constexpr const char kXfbVerticesPerInstance[] = "xfbVerticesPerInstance";
constexpr const char kCoverageMask[]           = "coverageMask";
constexpr const char kUnused[]                 = "unused";

}  // namespace

// class DriverUniformMetal
// The fields here must match the DriverUniforms structure defined in ContextMtl.h.
TFieldList *DriverUniformMetal::createUniformFields(TSymbolTable *symbolTable)
{
    TFieldList *driverFieldList = DriverUniform::createUniformFields(symbolTable);

    constexpr size_t kNumGraphicsDriverUniformsMetal = 4;
    constexpr std::array<const char *, kNumGraphicsDriverUniformsMetal>
        kGraphicsDriverUniformNamesMetal = {
            {kXfbBufferOffsets, kXfbVerticesPerInstance, kCoverageMask, kUnused}};

    const std::array<TType *, kNumGraphicsDriverUniformsMetal> kDriverUniformTypesMetal = {{
        // xfbBufferOffsets: uvec4
        new TType(EbtInt, EbpHigh, EvqGlobal, 4),
        // xfbVerticesPerInstance: uint
        new TType(EbtInt, EbpHigh, EvqGlobal),
        // coverageMask: uint
        new TType(EbtUInt, EbpHigh, EvqGlobal),
        // unused: uvec2
        new TType(EbtUInt, EbpHigh, EvqGlobal, 2),
    }};

    for (size_t uniformIndex = 0; uniformIndex < kNumGraphicsDriverUniformsMetal; ++uniformIndex)
    {
        TField *driverUniformField =
            new TField(kDriverUniformTypesMetal[uniformIndex],
                       ImmutableString(kGraphicsDriverUniformNamesMetal[uniformIndex]),
                       TSourceLoc(), SymbolType::AngleInternal);
        driverFieldList->push_back(driverUniformField);
    }

    return driverFieldList;
}

TIntermTyped *DriverUniformMetal::getCoverageMaskField() const
{
    return createDriverUniformRef(kCoverageMask);
}

}  // namespace sh
