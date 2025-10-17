/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
// Semaphore.h: Defines the gl::Semaphore class [EXT_external_objects]

#ifndef LIBANGLE_SEMAPHORE_H_
#define LIBANGLE_SEMAPHORE_H_

#include <memory>

#include "angle_gl.h"
#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/RefCountObject.h"
#include "libANGLE/angletypes.h"

namespace rx
{
class GLImplFactory;
class SemaphoreImpl;
}  // namespace rx

namespace gl
{
class Context;

class Semaphore final : public RefCountObject<SemaphoreID>
{
  public:
    Semaphore(rx::GLImplFactory *factory, SemaphoreID id);
    ~Semaphore() override;

    void onDestroy(const Context *context) override;

    rx::SemaphoreImpl *getImplementation() const { return mImplementation.get(); }

    angle::Result importFd(Context *context, HandleType handleType, GLint fd);
    angle::Result importZirconHandle(Context *context, HandleType handleType, GLuint handle);

    angle::Result wait(Context *context,
                       const BufferBarrierVector &bufferBarriers,
                       const TextureBarrierVector &textureBarriers);

    angle::Result signal(Context *context,
                         const BufferBarrierVector &bufferBarriers,
                         const TextureBarrierVector &textureBarriers);

  private:
    std::unique_ptr<rx::SemaphoreImpl> mImplementation;
};

}  // namespace gl

#endif  // LIBANGLE_SEMAPHORE_H_
