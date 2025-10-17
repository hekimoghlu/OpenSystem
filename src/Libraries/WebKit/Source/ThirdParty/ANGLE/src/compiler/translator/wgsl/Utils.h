/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_WGSL_UTILS_H_
#define COMPILER_TRANSLATOR_WGSL_UTILS_H_

#include "compiler/translator/Common.h"
#include "compiler/translator/InfoSink.h"
#include "compiler/translator/IntermNode.h"
#include "compiler/translator/Types.h"

namespace sh
{

// Can be used with TSymbol or TField or TFunc.
template <typename StringStreamType, typename Object>
void WriteNameOf(StringStreamType &output, const Object &namedObject)
{
    WriteNameOf(output, namedObject.symbolType(), namedObject.name());
}

template <typename StringStreamType>
void WriteNameOf(StringStreamType &output, SymbolType symbolType, const ImmutableString &name);

enum class WgslAddressSpace
{
    Uniform,
    NonUniform
};

struct EmitTypeConfig
{
    // If `addressSpace` is WgslAddressSpace::Uniform, all arrays with stride not a multiple of 16
    // will need a wrapper struct for the array element type that is of size a multiple of 16, if
    // the array element type that is not already a struct. This is to satisfy WGSL's uniform
    // address space layout constraints.
    WgslAddressSpace addressSpace = WgslAddressSpace::NonUniform;
};

template <typename StringStreamType>
void WriteWgslBareTypeName(StringStreamType &output,
                           const TType &type,
                           const EmitTypeConfig &config);
template <typename StringStreamType>
void WriteWgslType(StringStreamType &output, const TType &type, const EmitTypeConfig &config);

// From the type, creates a legal WGSL name for a struct that wraps it.
ImmutableString MakeUniformWrapperStructName(const TType *type);

// Returns true if a `type` in the uniform address space is an array that needs its element type
// wrapped in a struct.
bool ElementTypeNeedsUniformWrapperStruct(bool inUniformAddressSpace, const TType *type);

using GlobalVars = TMap<ImmutableString, TIntermDeclaration *>;
GlobalVars FindGlobalVars(TIntermBlock *root);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_WGSL_UTILS_H_
