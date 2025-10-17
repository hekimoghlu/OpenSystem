/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 1, 2022.
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
// DisplayVkLinux.h:
//    Defines the class interface for DisplayVkLinux, which is the base of DisplayVkSimple,
//    DisplayVkHeadless, DisplayVkXcb and DisplayVkWayland.  This base class implements the
//    common functionality of handling Linux dma-bufs.
//

#ifndef LIBANGLE_RENDERER_VULKAN_DISPLAY_DISPLAYVKLINUX_H_
#define LIBANGLE_RENDERER_VULKAN_DISPLAY_DISPLAYVKLINUX_H_

#include "libANGLE/renderer/vulkan/DisplayVk.h"

namespace rx
{
class DisplayVkLinux : public DisplayVk
{
  public:
    DisplayVkLinux(const egl::DisplayState &state);

    DeviceImpl *createDevice() override;

    ExternalImageSiblingImpl *createExternalImageSibling(const gl::Context *context,
                                                         EGLenum target,
                                                         EGLClientBuffer buffer,
                                                         const egl::AttributeMap &attribs) override;
    std::vector<VkDrmFormatModifierPropertiesEXT> GetDrmModifiers(const DisplayVk *displayVk,
                                                                  VkFormat vkFormat);
    bool SupportsDrmModifiers(VkPhysicalDevice device, VkFormat vkFormat);
    std::vector<VkFormat> GetVkFormatsWithDrmModifiers(const vk::Renderer *renderer);
    std::vector<EGLint> GetDrmFormats(const vk::Renderer *renderer);
    bool supportsDmaBufFormat(EGLint format) const override;
    egl::Error queryDmaBufFormats(EGLint maxFormats, EGLint *formats, EGLint *numFormats) override;
    egl::Error queryDmaBufModifiers(EGLint format,
                                    EGLint maxModifiers,
                                    EGLuint64KHR *modifiers,
                                    EGLBoolean *externalOnly,
                                    EGLint *numModifiers) override;

  private:
    // Supported DRM formats
    std::vector<EGLint> mDrmFormats;

    bool mDrmFormatsInitialized;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_DISPLAY_DISPLAYVKLINUX_H_
