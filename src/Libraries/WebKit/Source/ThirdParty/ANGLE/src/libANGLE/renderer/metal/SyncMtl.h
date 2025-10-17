/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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
// Copyright (c) 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SyncMtl:
//    Defines the class interface for SyncMtl, implementing SyncImpl.
//

#ifndef LIBANGLE_RENDERER_METAL_SYNCMTL_H_
#define LIBANGLE_RENDERER_METAL_SYNCMTL_H_

#include <optional>

#include "libANGLE/renderer/EGLSyncImpl.h"
#include "libANGLE/renderer/FenceNVImpl.h"
#include "libANGLE/renderer/SyncImpl.h"
#include "libANGLE/renderer/metal/mtl_common.h"

namespace egl
{
class AttributeMap;
}

namespace rx
{

class ContextMtl;

namespace mtl
{
class SyncImpl;
}  // namespace mtl

class FenceNVMtl : public FenceNVImpl
{
  public:
    FenceNVMtl();
    ~FenceNVMtl() override;
    void onDestroy(const gl::Context *context) override;
    angle::Result set(const gl::Context *context, GLenum condition) override;
    angle::Result test(const gl::Context *context, GLboolean *outFinished) override;
    angle::Result finish(const gl::Context *context) override;

  private:
    std::unique_ptr<mtl::SyncImpl> mSync;
};

class SyncMtl : public SyncImpl
{
  public:
    SyncMtl();
    ~SyncMtl() override;

    void onDestroy(const gl::Context *context) override;

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
    std::unique_ptr<mtl::SyncImpl> mSync;
};

class EGLSyncMtl final : public EGLSyncImpl
{
  public:
    EGLSyncMtl();
    ~EGLSyncMtl() override;

    void onDestroy(const egl::Display *display) override;

    egl::Error initialize(const egl::Display *display,
                          const gl::Context *context,
                          EGLenum type,
                          const egl::AttributeMap &attribs) override;
    egl::Error clientWait(const egl::Display *display,
                          const gl::Context *context,
                          EGLint flags,
                          EGLTime timeout,
                          EGLint *outResult) override;
    egl::Error serverWait(const egl::Display *display,
                          const gl::Context *context,
                          EGLint flags) override;
    egl::Error getStatus(const egl::Display *display, EGLint *outStatus) override;

    egl::Error copyMetalSharedEventANGLE(const egl::Display *display, void **result) const override;
    egl::Error dupNativeFenceFD(const egl::Display *display, EGLint *result) const override;

  private:
    mtl::AutoObjCPtr<id<MTLSharedEvent>> mSharedEvent;

    std::unique_ptr<mtl::SyncImpl> mSync;
};

}  // namespace rx

#endif
