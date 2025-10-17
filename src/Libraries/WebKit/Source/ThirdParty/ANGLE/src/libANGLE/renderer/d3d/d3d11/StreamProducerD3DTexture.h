/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// StreamProducerD3DTexture.h: Interface for a D3D11 texture stream producer

#ifndef LIBANGLE_RENDERER_D3D_D3D11_STREAM11_H_
#define LIBANGLE_RENDERER_D3D_D3D11_STREAM11_H_

#include "libANGLE/renderer/StreamProducerImpl.h"

namespace rx
{
class Renderer11;

class StreamProducerD3DTexture : public StreamProducerImpl
{
  public:
    StreamProducerD3DTexture(Renderer11 *renderer);
    ~StreamProducerD3DTexture() override;

    egl::Error validateD3DTexture(const void *pointer,
                                  const egl::AttributeMap &attributes) const override;
    void postD3DTexture(void *pointer, const egl::AttributeMap &attributes) override;
    egl::Stream::GLTextureDescription getGLFrameDescription(int planeIndex) override;

    // Gets a pointer to the internal D3D texture
    ID3D11Texture2D *getD3DTexture();

    // Gets the slice index for the D3D texture that the frame is in
    UINT getArraySlice();

  private:
    Renderer11 *mRenderer;

    ID3D11Texture2D *mTexture;
    UINT mArraySlice;
    UINT mPlaneOffset;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D11_STREAM11_H_
