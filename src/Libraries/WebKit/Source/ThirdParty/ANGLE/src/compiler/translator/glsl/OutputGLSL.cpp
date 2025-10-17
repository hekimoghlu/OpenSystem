/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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

#include "compiler/translator/glsl/OutputGLSL.h"

#include "compiler/translator/Compiler.h"

namespace sh
{

TOutputGLSL::TOutputGLSL(TCompiler *compiler,
                         TInfoSinkBase &objSink,
                         const ShCompileOptions &compileOptions)
    : TOutputGLSLBase(compiler, objSink, compileOptions)
{}

bool TOutputGLSL::writeVariablePrecision(TPrecision)
{
    return false;
}

void TOutputGLSL::visitSymbol(TIntermSymbol *node)
{
    TInfoSinkBase &out = objSink();

    // All the special cases are built-ins, so if it's not a built-in we can return early.
    if (node->variable().symbolType() != SymbolType::BuiltIn)
    {
        TOutputGLSLBase::visitSymbol(node);
        return;
    }

    // Some built-ins get a special translation.
    const ImmutableString &name = node->getName();
    if (name == "gl_FragDepthEXT")
    {
        out << "gl_FragDepth";
    }
    else if (name == "gl_FragColor" && sh::IsGLSL130OrNewer(getShaderOutput()))
    {
        out << "webgl_FragColor";
    }
    else if (name == "gl_FragData" && sh::IsGLSL130OrNewer(getShaderOutput()))
    {
        out << "webgl_FragData";
    }
    else if (name == "gl_SecondaryFragColorEXT")
    {
        out << "webgl_SecondaryFragColor";
    }
    else if (name == "gl_SecondaryFragDataEXT")
    {
        out << "webgl_SecondaryFragData";
    }
    else
    {
        TOutputGLSLBase::visitSymbol(node);
    }
}

ImmutableString TOutputGLSL::translateTextureFunction(const ImmutableString &name,
                                                      const ShCompileOptions &option)
{
    // Check WEBGL_video_texture invocation first.
    if (name == "textureVideoWEBGL")
    {
        if (option.takeVideoTextureAsExternalOES)
        {
            // TODO(http://anglebug.com/42262534): Implement external image situation.
            UNIMPLEMENTED();
            return ImmutableString("");
        }
        else
        {
            // Default translating textureVideoWEBGL to texture2D.
            return ImmutableString("texture2D");
        }
    }

    static const char *simpleRename[]       = {"texture2DLodEXT",
                                               "texture2DLod",
                                               "texture2DProjLodEXT",
                                               "texture2DProjLod",
                                               "textureCubeLodEXT",
                                               "textureCubeLod",
                                               "texture2DGradEXT",
                                               "texture2DGradARB",
                                               "texture2DProjGradEXT",
                                               "texture2DProjGradARB",
                                               "textureCubeGradEXT",
                                               "textureCubeGradARB",
                                               nullptr,
                                               nullptr};
    static const char *legacyToCoreRename[] = {
        "texture2D", "texture", "texture2DProj", "textureProj", "texture2DLod", "textureLod",
        "texture2DProjLod", "textureProjLod", "texture2DRect", "texture", "texture2DRectProj",
        "textureProj", "textureCube", "texture", "textureCubeLod", "textureLod",
        // Extensions
        "texture2DLodEXT", "textureLod", "texture2DProjLodEXT", "textureProjLod",
        "textureCubeLodEXT", "textureLod", "texture2DGradEXT", "textureGrad",
        "texture2DProjGradEXT", "textureProjGrad", "textureCubeGradEXT", "textureGrad", "texture3D",
        "texture", "texture3DProj", "textureProj", "texture3DLod", "textureLod", "texture3DProjLod",
        "textureProjLod", "shadow2DEXT", "texture", "shadow2DProjEXT", "textureProj", nullptr,
        nullptr};
    const char **mapping =
        (sh::IsGLSL130OrNewer(getShaderOutput())) ? legacyToCoreRename : simpleRename;

    for (int i = 0; mapping[i] != nullptr; i += 2)
    {
        if (name == mapping[i])
        {
            return ImmutableString(mapping[i + 1]);
        }
    }

    return name;
}

}  // namespace sh
