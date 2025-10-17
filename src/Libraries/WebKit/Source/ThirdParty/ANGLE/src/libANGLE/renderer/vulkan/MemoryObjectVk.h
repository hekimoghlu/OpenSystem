/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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

// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// MemoryObjectVk.h: Defines the class interface for MemoryObjectVk,
// implementing MemoryObjectImpl.

#ifndef LIBANGLE_RENDERER_VULKAN_MEMORYOBJECTVK_H_
#define LIBANGLE_RENDERER_VULKAN_MEMORYOBJECTVK_H_

#include "libANGLE/renderer/MemoryObjectImpl.h"
#include "libANGLE/renderer/vulkan/vk_helpers.h"
#include "libANGLE/renderer/vulkan/vk_wrapper.h"

namespace rx
{

class MemoryObjectVk : public MemoryObjectImpl
{
  public:
    MemoryObjectVk();
    ~MemoryObjectVk() override;

    void onDestroy(const gl::Context *context) override;

    angle::Result setDedicatedMemory(const gl::Context *context, bool dedicatedMemory) override;
    angle::Result setProtectedMemory(const gl::Context *context, bool protectedMemory) override;

    angle::Result importFd(gl::Context *context,
                           GLuint64 size,
                           gl::HandleType handleType,
                           GLint fd) override;

    angle::Result importZirconHandle(gl::Context *context,
                                     GLuint64 size,
                                     gl::HandleType handleType,
                                     GLuint handle) override;

    angle::Result createImage(ContextVk *context,
                              gl::TextureType type,
                              size_t levels,
                              GLenum internalFormat,
                              const gl::Extents &size,
                              GLuint64 offset,
                              vk::ImageHelper *image,
                              GLbitfield createFlags,
                              GLbitfield usageFlags,
                              const void *imageCreateInfoPNext);

  private:
    static constexpr int kInvalidFd = -1;
    angle::Result importOpaqueFd(ContextVk *contextVk, GLuint64 size, GLint fd);
    angle::Result importZirconVmo(ContextVk *contextVk, GLuint64 size, GLuint handle);

    // Imported memory object was a dedicated allocation.
    bool mDedicatedMemory = false;
    bool mProtectedMemory = false;

    GLuint64 mSize             = 0;
    gl::HandleType mHandleType = gl::HandleType::InvalidEnum;
    int mFd                    = kInvalidFd;

    zx_handle_t mZirconHandle = ZX_HANDLE_INVALID;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_MEMORYOBJECTVK_H_
