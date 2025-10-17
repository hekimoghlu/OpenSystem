/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ImageMtl.h:
//    Defines the class interface for ImageMtl, implementing ImageImpl.
//

#ifndef LIBANGLE_RENDERER_METAL_IMAGEMTL_H
#define LIBANGLE_RENDERER_METAL_IMAGEMTL_H

#include "libANGLE/renderer/ImageImpl.h"
#include "libANGLE/renderer/metal/mtl_resources.h"

namespace rx
{

class DisplayMtl;

class TextureImageSiblingMtl : public ExternalImageSiblingImpl
{
  public:
    TextureImageSiblingMtl(EGLClientBuffer buffer, const egl::AttributeMap &attribs);
    ~TextureImageSiblingMtl() override;

    static egl::Error ValidateClientBuffer(const DisplayMtl *display,
                                           EGLClientBuffer buffer,
                                           const egl::AttributeMap &attribs);

    egl::Error initialize(const egl::Display *display) override;
    void onDestroy(const egl::Display *display) override;

    // ExternalImageSiblingImpl interface
    gl::Format getFormat() const override;
    bool isRenderable(const gl::Context *context) const override;
    bool isTexturable(const gl::Context *context) const override;
    gl::Extents getSize() const override;
    size_t getSamples() const override;

    bool isYUV() const override;
    bool hasProtectedContent() const override;

    const mtl::TextureRef &getTexture() const { return mNativeTexture; }
    const mtl::Format &getFormatMtl() const { return mFormat; }

  private:
    angle::Result initImpl(DisplayMtl *display);

    EGLClientBuffer mBuffer;
    egl::AttributeMap mAttribs;
    gl::Format mGLFormat;
    mtl::Format mFormat;

    bool mRenderable  = false;
    bool mTextureable = false;

    mtl::TextureRef mNativeTexture;
};

class ImageMtl : public ImageImpl
{
  public:
    ImageMtl(const egl::ImageState &state, const gl::Context *context);
    ~ImageMtl() override;
    void onDestroy(const egl::Display *display) override;

    egl::Error initialize(const egl::Display *display) override;

    angle::Result orphan(const gl::Context *context, egl::ImageSibling *sibling) override;

    const mtl::TextureRef &getTexture() const { return mNativeTexture; }
    gl::TextureType getImageTextureType() const { return mImageTextureType; }
    uint32_t getImageLevel() const { return mImageLevel; }
    uint32_t getImageLayer() const { return mImageLayer; }

  private:
    gl::TextureType mImageTextureType;
    uint32_t mImageLevel = 0;
    uint32_t mImageLayer = 0;

    mtl::TextureRef mNativeTexture;
};
}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_IMAGEMTL_H */
