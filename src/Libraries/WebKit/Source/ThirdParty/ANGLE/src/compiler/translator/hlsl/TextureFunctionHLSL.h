/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 10, 2024.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// TextureFunctionHLSL: Class for writing implementations of ESSL texture functions into HLSL
// output. Some of the implementations are straightforward and just call the HLSL equivalent of the
// ESSL texture function, others do more work to emulate ESSL texture sampling or size query
// behavior.
//

#ifndef COMPILER_TRANSLATOR_HLSL_TEXTUREFUNCTIONHLSL_H_
#define COMPILER_TRANSLATOR_HLSL_TEXTUREFUNCTIONHLSL_H_

#include <set>

#include "GLSLANG/ShaderLang.h"
#include "compiler/translator/BaseTypes.h"
#include "compiler/translator/Common.h"
#include "compiler/translator/InfoSink.h"

namespace sh
{

class TextureFunctionHLSL final : angle::NonCopyable
{
  public:
    struct TextureFunction
    {
        // See ESSL 3.00.6 section 8.8 for reference about what the different methods below do.
        enum Method
        {
            IMPLICIT,  // Mipmap LOD determined implicitly (standard lookup)
            BIAS,
            LOD,
            LOD0,
            LOD0BIAS,
            SIZE,  // textureSize()
            FETCH,
            GRAD,
            GATHER
        };

        ImmutableString name() const;

        bool operator<(const TextureFunction &rhs) const;

        const char *getReturnType() const;

        TBasicType sampler;
        int coords;
        bool proj;
        bool offset;
        Method method;
    };

    // Returns the name of the texture function implementation to call.
    // The name that's passed in is the name of the GLSL texture function that it should implement.
    ImmutableString useTextureFunction(const ImmutableString &name,
                                       TBasicType samplerType,
                                       int coords,
                                       size_t argumentCount,
                                       bool lod0,
                                       sh::GLenum shaderType);

    void textureFunctionHeader(TInfoSinkBase &out,
                               const ShShaderOutput outputType,
                               bool getDimensionsIgnoresBaseLevel);

  private:
    typedef std::set<TextureFunction> TextureFunctionSet;
    TextureFunctionSet mUsesTexture;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_HLSL_TEXTUREFUNCTIONHLSL_H_
