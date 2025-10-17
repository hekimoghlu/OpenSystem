/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// CompilerGL.h: Defines the class interface for CompilerGL.

#ifndef LIBANGLE_RENDERER_GL_COMPILERGL_H_
#define LIBANGLE_RENDERER_GL_COMPILERGL_H_

#include "libANGLE/renderer/CompilerImpl.h"

namespace rx
{
class ContextGL;

class CompilerGL : public CompilerImpl
{
  public:
    CompilerGL(const ContextGL *);
    ~CompilerGL() override {}

    ShShaderOutput getTranslatorOutputType() const override;

  private:
    ShShaderOutput mTranslatorOutputType;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_COMPILERGL_H_
