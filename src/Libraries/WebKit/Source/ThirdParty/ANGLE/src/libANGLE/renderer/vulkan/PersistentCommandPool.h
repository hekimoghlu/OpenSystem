/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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
// PersistentCommandPool.h:
//    Defines the class interface for PersistentCommandBuffer
//

#ifndef LIBANGLE_RENDERER_VULKAN_PERSISTENTCOMMANDPOOL_H_
#define LIBANGLE_RENDERER_VULKAN_PERSISTENTCOMMANDPOOL_H_

#include <vector>

#include "libANGLE/renderer/vulkan/vk_utils.h"
#include "libANGLE/renderer/vulkan/vk_wrapper.h"

namespace rx
{

namespace vk
{

class PersistentCommandPool final
{
  public:
    PersistentCommandPool();
    ~PersistentCommandPool();

    void destroy(VkDevice device);
    angle::Result init(ErrorContext *context,
                       ProtectionType protectionType,
                       uint32_t queueFamilyIndex);

    angle::Result allocate(ErrorContext *context, PrimaryCommandBuffer *commandBufferOut);
    angle::Result collect(ErrorContext *context, PrimaryCommandBuffer &&buffer);

    bool valid() const { return mCommandPool.valid(); }

  private:
    angle::Result allocateCommandBuffer(ErrorContext *context);

    std::vector<PrimaryCommandBuffer> mFreeBuffers;

    CommandPool mCommandPool;

    static const int kInitBufferNum = 2;
};

}  // namespace vk

}  // namespace rx

#endif
