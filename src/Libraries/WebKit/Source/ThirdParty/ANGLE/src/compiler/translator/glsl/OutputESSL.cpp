/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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

#include "compiler/translator/glsl/OutputESSL.h"

namespace sh
{

TOutputESSL::TOutputESSL(TCompiler *compiler,
                         TInfoSinkBase &objSink,
                         const ShCompileOptions &compileOptions)
    : TOutputGLSLBase(compiler, objSink, compileOptions)
{}

bool TOutputESSL::writeVariablePrecision(TPrecision precision)
{
    if (precision == EbpUndefined)
        return false;

    if (precision == EbpHigh && !isHighPrecisionSupported())
    {
        precision = EbpMedium;
    }

    TInfoSinkBase &out = objSink();
    out << getPrecisionString(precision);
    return true;
}

ImmutableString TOutputESSL::translateTextureFunction(const ImmutableString &name,
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

    return name;
}

}  // namespace sh
