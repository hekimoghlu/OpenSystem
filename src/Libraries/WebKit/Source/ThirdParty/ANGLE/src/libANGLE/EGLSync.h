/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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

// EGLSync.h: Defines the egl::Sync classes, which support the EGL_KHR_fence_sync,
// EGL_KHR_wait_sync and EGL 1.5 sync objects.

#ifndef LIBANGLE_EGLSYNC_H_
#define LIBANGLE_EGLSYNC_H_

#include "libANGLE/Debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/RefCountObject.h"

#include "common/angleutils.h"

namespace rx
{
class EGLImplFactory;
class EGLSyncImpl;
}  // namespace rx

namespace gl
{
class Context;
}  // namespace gl

namespace egl
{
class Sync final : public LabeledObject
{
  public:
    Sync(rx::EGLImplFactory *factory, EGLenum type);
    ~Sync() override;

    void setLabel(EGLLabelKHR label) override;
    EGLLabelKHR getLabel() const override;

    const SyncID &id() const { return mId; }

    void onDestroy(const Display *display);

    Error initialize(const Display *display,
                     const gl::Context *context,
                     const SyncID &id,
                     const AttributeMap &attribs);
    Error clientWait(const Display *display,
                     const gl::Context *context,
                     EGLint flags,
                     EGLTime timeout,
                     EGLint *outResult);
    Error serverWait(const Display *display, const gl::Context *context, EGLint flags);
    Error signal(const Display *display, const gl::Context *context, EGLint mode);
    Error getStatus(const Display *display, EGLint *outStatus) const;

    Error copyMetalSharedEventANGLE(const Display *display, void **result) const;
    Error dupNativeFenceFD(const Display *display, EGLint *result) const;

    EGLenum getType() const { return mType; }
    const AttributeMap &getAttributeMap() const { return mAttributeMap; }
    EGLint getCondition() const { return mCondition; }
    EGLint getNativeFenceFD() const { return mNativeFenceFD; }

  private:
    std::unique_ptr<rx::EGLSyncImpl> mFence;

    EGLLabelKHR mLabel;

    SyncID mId;
    EGLenum mType;
    AttributeMap mAttributeMap;
    EGLint mCondition;
    EGLint mNativeFenceFD;
};

}  // namespace egl

#endif  // LIBANGLE_FENCE_H_
