/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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

// ImageImpl.h: Defines the rx::ImageImpl class representing the EGLimage object.

#ifndef LIBANGLE_RENDERER_IMAGEIMPL_H_
#define LIBANGLE_RENDERER_IMAGEIMPL_H_

#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/formatutils.h"
#include "libANGLE/renderer/FramebufferAttachmentObjectImpl.h"

namespace gl
{
class Context;
}  // namespace gl

namespace egl
{
class Display;
class ImageSibling;
struct ImageState;
}  // namespace egl

namespace rx
{
class ExternalImageSiblingImpl : public FramebufferAttachmentObjectImpl
{
  public:
    ~ExternalImageSiblingImpl() override {}

    virtual egl::Error initialize(const egl::Display *display) = 0;
    virtual void onDestroy(const egl::Display *display) {}

    virtual gl::Format getFormat() const                        = 0;
    virtual bool isRenderable(const gl::Context *context) const = 0;
    virtual bool isTexturable(const gl::Context *context) const = 0;
    virtual bool isYUV() const                                  = 0;
    virtual bool hasFrontBufferUsage() const;
    virtual bool isCubeMap() const;
    virtual bool hasProtectedContent() const = 0;
    virtual gl::Extents getSize() const      = 0;
    virtual size_t getSamples() const        = 0;
    virtual uint32_t getLevelCount() const;
};

class ImageImpl : angle::NonCopyable
{
  public:
    ImageImpl(const egl::ImageState &state) : mState(state) {}
    virtual ~ImageImpl() {}
    virtual void onDestroy(const egl::Display *display) {}

    virtual egl::Error initialize(const egl::Display *display) = 0;

    virtual angle::Result orphan(const gl::Context *context, egl::ImageSibling *sibling) = 0;

    virtual egl::Error exportVkImage(void *vkImage, void *vkImageCreateInfo);

    virtual bool isFixedRatedCompression(const gl::Context *context);

  protected:
    const egl::ImageState &mState;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_IMAGEIMPL_H_
