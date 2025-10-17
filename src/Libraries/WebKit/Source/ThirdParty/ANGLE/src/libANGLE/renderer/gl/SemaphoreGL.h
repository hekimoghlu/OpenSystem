/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SempahoreGL.h: Defines the rx::SempahoreGL class, an implementation of SemaphoreImpl.

#ifndef LIBANGLE_RENDERER_GL_SEMAPHOREGL_H_
#define LIBANGLE_RENDERER_GL_SEMAPHOREGL_H_

#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/SemaphoreImpl.h"

namespace rx
{
class SemaphoreGL : public SemaphoreImpl
{
  public:
    SemaphoreGL(GLuint semaphoreID);
    ~SemaphoreGL() override;

    void onDestroy(const gl::Context *context) override;

    angle::Result importFd(gl::Context *context, gl::HandleType handleType, GLint fd) override;

    angle::Result importZirconHandle(gl::Context *context,
                                     gl::HandleType handleType,
                                     GLuint handle) override;

    angle::Result wait(gl::Context *context,
                       const gl::BufferBarrierVector &bufferBarriers,
                       const gl::TextureBarrierVector &textureBarriers) override;

    angle::Result signal(gl::Context *context,
                         const gl::BufferBarrierVector &bufferBarriers,
                         const gl::TextureBarrierVector &textureBarriers) override;

    GLuint getSemaphoreID() const;

  private:
    GLuint mSemaphoreID;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_SEMAPHOREGL_H_
