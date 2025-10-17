/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifdef ANGLE_ENABLE_NULL
#    include "compiler/translator/null/TranslatorNULL.h"
#endif  // ANGLE_ENABLE_NULL

#ifdef ANGLE_ENABLE_ESSL
#    include "compiler/translator/glsl/TranslatorESSL.h"
#endif  // ANGLE_ENABLE_ESSL

#ifdef ANGLE_ENABLE_GLSL
#    include "compiler/translator/glsl/TranslatorGLSL.h"
#endif  // ANGLE_ENABLE_GLSL

#ifdef ANGLE_ENABLE_HLSL
#    include "compiler/translator/hlsl/TranslatorHLSL.h"
#endif  // ANGLE_ENABLE_HLSL

#ifdef ANGLE_ENABLE_VULKAN
#    include "compiler/translator/spirv/TranslatorSPIRV.h"
#endif  // ANGLE_ENABLE_VULKAN

#ifdef ANGLE_ENABLE_METAL
#    include "compiler/translator/msl/TranslatorMSL.h"
#endif  // ANGLE_ENABLE_METAL

#ifdef ANGLE_ENABLE_WGPU
#    include "compiler/translator/wgsl/TranslatorWGSL.h"
#endif  // ANGLE_ENABLE_WGPU

#include "compiler/translator/util.h"

namespace sh
{

//
// This function must be provided to create the actual
// compile object used by higher level code.  It returns
// a subclass of TCompiler.
//
TCompiler *ConstructCompiler(sh::GLenum type, ShShaderSpec spec, ShShaderOutput output)
{
#ifdef ANGLE_ENABLE_NULL
    if (IsOutputNULL(output))
    {
        return new TranslatorNULL(type, spec);
    }
#endif  // ANGLE_ENABLE_NULL

#ifdef ANGLE_ENABLE_ESSL
    if (IsOutputESSL(output))
    {
        return new TranslatorESSL(type, spec);
    }
#endif  // ANGLE_ENABLE_ESSL

#ifdef ANGLE_ENABLE_GLSL
    if (IsOutputGLSL(output))
    {
        return new TranslatorGLSL(type, spec, output);
    }
#endif  // ANGLE_ENABLE_GLSL

#ifdef ANGLE_ENABLE_HLSL
    if (IsOutputHLSL(output))
    {
        return new TranslatorHLSL(type, spec, output);
    }
#endif  // ANGLE_ENABLE_HLSL

#ifdef ANGLE_ENABLE_VULKAN
    if (IsOutputSPIRV(output))
    {
        return new TranslatorSPIRV(type, spec);
    }
#endif  // ANGLE_ENABLE_VULKAN

#ifdef ANGLE_ENABLE_METAL
    if (IsOutputMSL(output))
    {
        return new TranslatorMSL(type, spec, output);
    }
#endif  // ANGLE_ENABLE_METAL

#ifdef ANGLE_ENABLE_WGPU
    if (IsOutputWGSL(output))
    {
        return new TranslatorWGSL(type, spec, output);
    }
#endif  // ANGLE_ENABLE_WGPU

    // Unsupported compiler or unknown format. Return nullptr per the sh::ConstructCompiler API.
    return nullptr;
}

//
// Delete the compiler made by ConstructCompiler
//
void DeleteCompiler(TCompiler *compiler)
{
    SafeDelete(compiler);
}

}  // namespace sh
