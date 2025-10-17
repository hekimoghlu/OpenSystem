/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 22, 2024.
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

// ShaderGL.h: Defines the class interface for ShaderGL.

#ifndef LIBANGLE_RENDERER_GL_SHADERGL_H_
#define LIBANGLE_RENDERER_GL_SHADERGL_H_

#include "libANGLE/renderer/ShaderImpl.h"

namespace rx
{
class RendererGL;
enum class MultiviewImplementationTypeGL;

class ShaderGL : public ShaderImpl
{
  public:
    ShaderGL(const gl::ShaderState &state, GLuint shaderID);
    ~ShaderGL() override;

    void onDestroy(const gl::Context *context) override;

    std::shared_ptr<ShaderTranslateTask> compile(const gl::Context *context,
                                                 ShCompileOptions *options) override;
    std::shared_ptr<ShaderTranslateTask> load(const gl::Context *context,
                                              gl::BinaryInputStream *stream) override;

    std::string getDebugInfo() const override;

    GLuint getShaderID() const;

  private:
    GLuint mShaderID = 0;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_SHADERGL_H_
