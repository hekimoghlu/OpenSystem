/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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
// vk_command_buffer_utils:
//    Helpers for secondary command buffer implementations.
//

#ifndef LIBANGLE_RENDERER_VULKAN_VK_COMMAND_BUFFER_UTILS_H_
#define LIBANGLE_RENDERER_VULKAN_VK_COMMAND_BUFFER_UTILS_H_

#include "common/PackedEnums.h"
#include "common/angleutils.h"

namespace rx
{
namespace vk
{

enum class ProtectionType : uint8_t
{
    Unprotected = 0,
    Protected   = 1,

    InvalidEnum = 2,
    EnumCount   = 2,
};

using ProtectionTypes = angle::PackedEnumBitSet<ProtectionType, uint8_t>;

ANGLE_INLINE ProtectionType ConvertProtectionBoolToType(bool isProtected)
{
    return (isProtected ? ProtectionType::Protected : ProtectionType::Unprotected);
}

// A helper class to track commands recorded to a command buffer.
class CommandBufferCommandTracker
{
  public:
    void onDraw() { ++mRenderPassWriteCommandCount; }
    void onClearAttachments() { ++mRenderPassWriteCommandCount; }
    uint32_t getRenderPassWriteCommandCount() const { return mRenderPassWriteCommandCount; }

    void reset() { *this = CommandBufferCommandTracker{}; }

  private:
    // The number of commands recorded that can modify a render pass attachment, i.e.
    // vkCmdClearAttachment and vkCmdDraw*.  Used to know if a command might have written to an
    // attachment after it was invalidated.
    uint32_t mRenderPassWriteCommandCount = 0;
};

}  // namespace vk
}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_VK_COMMAND_BUFFER_UTILS_H_
