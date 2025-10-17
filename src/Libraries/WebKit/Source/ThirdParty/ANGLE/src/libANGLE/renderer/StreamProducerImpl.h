/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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

// StreamProducerImpl.h: Defines the abstract rx::StreamProducerImpl class.

#ifndef LIBANGLE_RENDERER_STREAMPRODUCERIMPL_H_
#define LIBANGLE_RENDERER_STREAMPRODUCERIMPL_H_

#include "common/angleutils.h"
#include "libANGLE/Stream.h"

namespace rx
{

class StreamProducerImpl : angle::NonCopyable
{
  public:
    explicit StreamProducerImpl() {}
    virtual ~StreamProducerImpl() {}

    // Validates the ability for the producer to accept an arbitrary pointer to a frame. All
    // pointers should be validated through this function before being used to produce a frame.
    virtual egl::Error validateD3DTexture(const void *pointer,
                                          const egl::AttributeMap &attributes) const = 0;

    // Constructs a frame from an arbitrary external pointer that points to producer specific frame
    // data. Replaces the internal frame with the new one.
    virtual void postD3DTexture(void *pointer, const egl::AttributeMap &attributes) = 0;

    // Returns an OpenGL texture interpretation of some frame attributes for the purpose of
    // constructing an OpenGL texture from a frame. Depending on the producer and consumer, some
    // frames may have multiple "planes" with different OpenGL texture representations.
    virtual egl::Stream::GLTextureDescription getGLFrameDescription(int planeIndex) = 0;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_STREAMPRODUCERIMPL_H_
