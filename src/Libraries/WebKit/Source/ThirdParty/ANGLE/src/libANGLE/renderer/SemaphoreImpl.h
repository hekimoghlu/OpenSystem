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
// SemaphoreImpl.h: Implements the rx::SemaphoreImpl class [EXT_external_objects]

#ifndef LIBANGLE_RENDERER_SEMAPHOREIMPL_H_
#define LIBANGLE_RENDERER_SEMAPHOREIMPL_H_

#include "angle_gl.h"
#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/angletypes.h"

namespace gl
{
class Context;
class Semaphore;
}  // namespace gl

namespace rx
{

class SemaphoreImpl : angle::NonCopyable
{
  public:
    virtual ~SemaphoreImpl() {}

    virtual void onDestroy(const gl::Context *context) = 0;

    virtual angle::Result importFd(gl::Context *context, gl::HandleType handleType, GLint fd) = 0;

    virtual angle::Result importZirconHandle(gl::Context *context,
                                             gl::HandleType handleType,
                                             GLuint handle) = 0;

    virtual angle::Result wait(gl::Context *context,
                               const gl::BufferBarrierVector &bufferBarriers,
                               const gl::TextureBarrierVector &textureBarriers) = 0;

    virtual angle::Result signal(gl::Context *context,
                                 const gl::BufferBarrierVector &bufferBarriers,
                                 const gl::TextureBarrierVector &textureBarriers) = 0;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_SEMAPHOREIMPL_H_
