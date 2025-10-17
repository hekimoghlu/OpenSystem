/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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

// Fence.h: Defines the gl::FenceNV and gl::Sync classes, which support the GL_NV_fence
// extension and GLES3 sync objects.

#ifndef LIBANGLE_FENCE_H_
#define LIBANGLE_FENCE_H_

#include "libANGLE/Debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/RefCountObject.h"

#include "common/angleutils.h"

namespace rx
{
class GLImplFactory;
class FenceNVImpl;
class SyncImpl;
}  // namespace rx

namespace gl
{

class FenceNV final : angle::NonCopyable
{
  public:
    explicit FenceNV(rx::GLImplFactory *factory);
    virtual ~FenceNV();

    void onDestroy(const gl::Context *context);
    angle::Result set(const Context *context, GLenum condition);
    angle::Result test(const Context *context, GLboolean *outResult);
    angle::Result finish(const Context *context);

    bool isSet() const { return mIsSet; }
    GLboolean getStatus() const { return mStatus; }
    GLenum getCondition() const { return mCondition; }

  private:
    rx::FenceNVImpl *mFence;

    bool mIsSet;

    GLboolean mStatus;
    GLenum mCondition;
};

class Sync final : public RefCountObject<SyncID>, public LabeledObject
{
  public:
    Sync(rx::GLImplFactory *factory, SyncID id);
    ~Sync() override;

    void onDestroy(const Context *context) override;

    angle::Result setLabel(const Context *context, const std::string &label) override;
    const std::string &getLabel() const override;

    angle::Result set(const Context *context, GLenum condition, GLbitfield flags);
    angle::Result clientWait(const Context *context,
                             GLbitfield flags,
                             GLuint64 timeout,
                             GLenum *outResult);
    angle::Result serverWait(const Context *context, GLbitfield flags, GLuint64 timeout);
    angle::Result getStatus(const Context *context, GLint *outResult) const;

    GLenum getCondition() const { return mCondition; }
    GLbitfield getFlags() const { return mFlags; }

    rx::SyncImpl *getImplementation() const { return mFence; }

  private:
    rx::SyncImpl *mFence;

    std::string mLabel;

    GLenum mCondition;
    GLbitfield mFlags;
};

}  // namespace gl

#endif  // LIBANGLE_FENCE_H_
