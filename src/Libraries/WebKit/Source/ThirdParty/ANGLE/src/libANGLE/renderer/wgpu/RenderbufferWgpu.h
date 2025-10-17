/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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
// RenderbufferWgpu.h:
//    Defines the class interface for RenderbufferWgpu, implementing RenderbufferImpl.
//

#ifndef LIBANGLE_RENDERER_WGPU_RENDERBUFFERWGPU_H_
#define LIBANGLE_RENDERER_WGPU_RENDERBUFFERWGPU_H_

#include "libANGLE/renderer/RenderbufferImpl.h"

namespace rx
{

class RenderbufferWgpu : public RenderbufferImpl
{
  public:
    RenderbufferWgpu(const gl::RenderbufferState &state);
    ~RenderbufferWgpu() override;

    angle::Result setStorage(const gl::Context *context,
                             GLenum internalformat,
                             GLsizei width,
                             GLsizei height) override;
    angle::Result setStorageMultisample(const gl::Context *context,
                                        GLsizei samples,
                                        GLenum internalformat,
                                        GLsizei width,
                                        GLsizei height,
                                        gl::MultisamplingMode mode) override;

    angle::Result setStorageEGLImageTarget(const gl::Context *context, egl::Image *image) override;

    angle::Result initializeContents(const gl::Context *context,
                                     GLenum binding,
                                     const gl::ImageIndex &imageIndex) override;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_WGPU_RENDERBUFFERWGPU_H_
