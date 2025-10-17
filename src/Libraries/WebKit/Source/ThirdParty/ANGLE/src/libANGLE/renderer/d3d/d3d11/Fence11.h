/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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

// Fence11.h: Defines the rx::FenceNV11 and rx::Sync11 classes which implement rx::FenceNVImpl
// and rx::SyncImpl.

#ifndef LIBANGLE_RENDERER_D3D_D3D11_FENCE11_H_
#define LIBANGLE_RENDERER_D3D_D3D11_FENCE11_H_

#include "libANGLE/renderer/FenceNVImpl.h"
#include "libANGLE/renderer/SyncImpl.h"

namespace rx
{
class Renderer11;

class FenceNV11 : public FenceNVImpl
{
  public:
    explicit FenceNV11(Renderer11 *renderer);
    ~FenceNV11() override;

    void onDestroy(const gl::Context *context) override {}
    angle::Result set(const gl::Context *context, GLenum condition) override;
    angle::Result test(const gl::Context *context, GLboolean *outFinished) override;
    angle::Result finish(const gl::Context *context) override;

  private:
    template <class T>
    friend angle::Result FenceSetHelper(const gl::Context *context, T *fence);
    template <class T>
    friend angle::Result FenceTestHelper(const gl::Context *context,
                                         T *fence,
                                         bool flushCommandBuffer,
                                         GLboolean *outFinished);

    Renderer11 *mRenderer;
    ID3D11Query *mQuery;
};

class Sync11 : public SyncImpl
{
  public:
    explicit Sync11(Renderer11 *renderer);
    ~Sync11() override;

    angle::Result set(const gl::Context *context, GLenum condition, GLbitfield flags) override;
    angle::Result clientWait(const gl::Context *context,
                             GLbitfield flags,
                             GLuint64 timeout,
                             GLenum *outResult) override;
    angle::Result serverWait(const gl::Context *context,
                             GLbitfield flags,
                             GLuint64 timeout) override;
    angle::Result getStatus(const gl::Context *context, GLint *outResult) override;

  private:
    template <class T>
    friend angle::Result FenceSetHelper(const gl::Context *context, T *fence);
    template <class T>
    friend angle::Result FenceTestHelper(const gl::Context *context,
                                         T *fence,
                                         bool flushCommandBuffer,
                                         GLboolean *outFinished);

    Renderer11 *mRenderer;
    ID3D11Query *mQuery;
    LONGLONG mCounterFrequency;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D11_FENCE11_H_
