/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ImageFunctionHLSL: Class for writing implementations of ESSL image functions into HLSL output.
//

#ifndef COMPILER_TRANSLATOR_HLSL_IMAGEFUNCTIONHLSL_H_
#define COMPILER_TRANSLATOR_HLSL_IMAGEFUNCTIONHLSL_H_

#include <set>

#include "GLSLANG/ShaderLang.h"
#include "compiler/translator/BaseTypes.h"
#include "compiler/translator/Common.h"
#include "compiler/translator/InfoSink.h"
#include "compiler/translator/Types.h"

namespace sh
{

class ImageFunctionHLSL final : angle::NonCopyable
{
  public:
    // Returns the name of the image function implementation to caller.
    // The name that's passed in is the name of the GLSL image function that it should implement.
    ImmutableString useImageFunction(const ImmutableString &name,
                                     const TBasicType &type,
                                     TLayoutImageInternalFormat imageInternalFormat,
                                     bool readonly);

    void imageFunctionHeader(TInfoSinkBase &out);
    const std::set<std::string> &getUsedImage2DFunctionNames() const
    {
        return mUsedImage2DFunctionNames;
    }

  private:
    struct ImageFunction
    {
        // See ESSL 3.10.4 section 8.12 for reference about what the different methods below do.
        enum class Method
        {
            SIZE,
            LOAD,
            STORE
        };

        enum class DataType
        {
            NONE,
            FLOAT4,
            UINT4,
            INT4,
            UNORM_FLOAT4,
            SNORM_FLOAT4
        };

        ImmutableString name() const;

        bool operator<(const ImageFunction &rhs) const;

        DataType getDataType(TLayoutImageInternalFormat format) const;

        const char *getReturnType() const;

        TBasicType image;
        TLayoutImageInternalFormat imageInternalFormat;
        bool readonly;
        Method method;
        DataType type;
    };

    static ImmutableString GetImageReference(TInfoSinkBase &out,
                                             const ImageFunctionHLSL::ImageFunction &imageFunction);
    static void OutputImageFunctionArgumentList(
        TInfoSinkBase &out,
        const ImageFunctionHLSL::ImageFunction &imageFunction);
    static void OutputImageSizeFunctionBody(TInfoSinkBase &out,
                                            const ImageFunctionHLSL::ImageFunction &imageFunction,
                                            const ImmutableString &imageReference);
    static void OutputImageLoadFunctionBody(TInfoSinkBase &out,
                                            const ImageFunctionHLSL::ImageFunction &imageFunction,
                                            const ImmutableString &imageReference);
    static void OutputImageStoreFunctionBody(TInfoSinkBase &out,
                                             const ImageFunctionHLSL::ImageFunction &imageFunction,
                                             const ImmutableString &imageReference);
    using ImageFunctionSet = std::set<ImageFunction>;
    ImageFunctionSet mUsesImage;
    std::set<std::string> mUsedImage2DFunctionNames;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_HLSL_IMAGEFUNCTIONHLSL_H_
