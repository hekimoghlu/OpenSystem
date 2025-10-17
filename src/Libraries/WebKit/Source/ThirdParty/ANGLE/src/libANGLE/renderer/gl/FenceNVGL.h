/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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

// FenceNVGL.h: Defines the class interface for FenceNVGL.

#ifndef LIBANGLE_RENDERER_GL_FENCENVGL_H_
#define LIBANGLE_RENDERER_GL_FENCENVGL_H_

#include "libANGLE/renderer/FenceNVImpl.h"

namespace rx
{
class FunctionsGL;

// FenceNV implemented with the native GL_NV_fence extension
class FenceNVGL : public FenceNVImpl
{
  public:
    explicit FenceNVGL(const FunctionsGL *functions);
    ~FenceNVGL() override;

    void onDestroy(const gl::Context *context) override {}
    angle::Result set(const gl::Context *context, GLenum condition) override;
    angle::Result test(const gl::Context *context, GLboolean *outFinished) override;
    angle::Result finish(const gl::Context *context) override;

    static bool Supported(const FunctionsGL *functions);

  private:
    GLuint mFence;

    const FunctionsGL *mFunctions;
};

// FenceNV implemented with the GLsync API
class FenceNVSyncGL : public FenceNVImpl
{
  public:
    explicit FenceNVSyncGL(const FunctionsGL *functions);
    ~FenceNVSyncGL() override;

    void onDestroy(const gl::Context *context) override {}
    angle::Result set(const gl::Context *context, GLenum condition) override;
    angle::Result test(const gl::Context *context, GLboolean *outFinished) override;
    angle::Result finish(const gl::Context *context) override;

    static bool Supported(const FunctionsGL *functions);

  private:
    GLsync mSyncObject;

    const FunctionsGL *mFunctions;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_FENCENVGL_H_
