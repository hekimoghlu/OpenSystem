/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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

// DisplayGL.h: Defines the class interface for DisplayGL.

#ifndef LIBANGLE_RENDERER_GL_DISPLAYGL_H_
#define LIBANGLE_RENDERER_GL_DISPLAYGL_H_

#include "libANGLE/renderer/DisplayImpl.h"
#include "libANGLE/renderer/ShareGroupImpl.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"

namespace egl
{
class Surface;
}

namespace rx
{

class ShareGroupGL : public ShareGroupImpl
{
  public:
    ShareGroupGL(const egl::ShareGroupState &state) : ShareGroupImpl(state) {}
};

class RendererGL;

class DisplayGL : public DisplayImpl
{
  public:
    DisplayGL(const egl::DisplayState &state);
    ~DisplayGL() override;

    egl::Error initialize(egl::Display *display) override;
    void terminate() override;

    ImageImpl *createImage(const egl::ImageState &state,
                           const gl::Context *context,
                           EGLenum target,
                           const egl::AttributeMap &attribs) override;

    SurfaceImpl *createPbufferFromClientBuffer(const egl::SurfaceState &state,
                                               EGLenum buftype,
                                               EGLClientBuffer clientBuffer,
                                               const egl::AttributeMap &attribs) override;

    StreamProducerImpl *createStreamProducerD3DTexture(egl::Stream::ConsumerType consumerType,
                                                       const egl::AttributeMap &attribs) override;

    ShareGroupImpl *createShareGroup(const egl::ShareGroupState &state) override;

    egl::Error makeCurrent(egl::Display *display,
                           egl::Surface *drawSurface,
                           egl::Surface *readSurface,
                           gl::Context *context) override;

    gl::Version getMaxConformantESVersion() const override;

    virtual RendererGL *getRenderer() const = 0;

    std::string getRendererDescription() override;
    std::string getVendorString() override;
    std::string getVersionString(bool includeFullVersion) override;

  protected:
    void generateExtensions(egl::DisplayExtensions *outExtensions) const override;

  private:
    virtual egl::Error makeCurrentSurfaceless(gl::Context *context);
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_DISPLAYGL_H_
