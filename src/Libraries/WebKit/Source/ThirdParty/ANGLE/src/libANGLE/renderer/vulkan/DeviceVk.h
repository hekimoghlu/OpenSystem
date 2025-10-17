/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 27, 2023.
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
// DeviceVk.h:
//    Defines the class interface for DeviceVk, implementing DeviceImpl.
//

#ifndef LIBANGLE_RENDERER_VULKAN_DEVICEVK_H_
#define LIBANGLE_RENDERER_VULKAN_DEVICEVK_H_

#include "libANGLE/renderer/DeviceImpl.h"

#include "common/vulkan/vk_headers.h"

namespace rx
{
namespace vk
{
class Renderer;
}

class DeviceVk : public DeviceImpl
{
  public:
    DeviceVk();
    ~DeviceVk() override;

    egl::Error initialize() override;
    egl::Error getAttribute(const egl::Display *display,
                            EGLint attribute,
                            void **outValue) override;
    void generateExtensions(egl::DeviceExtensions *outExtensions) const override;
    vk::Renderer *getRenderer() const { return mRenderer; }

  private:
    // Wrappers for some global vulkan methods which need to read env variables.
    // The wrappers will set those env variables before calling those global methods.
    static VKAPI_ATTR VkResult VKAPI_CALL
    WrappedCreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkInstance *pInstance);
    static VKAPI_ATTR VkResult VKAPI_CALL
    WrappedEnumerateInstanceExtensionProperties(const char *pLayerName,
                                                uint32_t *pPropertyCount,
                                                VkExtensionProperties *pProperties);
    static VKAPI_ATTR VkResult VKAPI_CALL
    WrappedEnumerateInstanceLayerProperties(uint32_t *pPropertyCount,
                                            VkLayerProperties *pProperties);
    static VKAPI_ATTR VkResult VKAPI_CALL WrappedEnumerateInstanceVersion(uint32_t *pApiVersion);
    static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL WrappedGetInstanceProcAddr(VkInstance instance,
                                                                               const char *pName);

    vk::Renderer *mRenderer = nullptr;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_DEVICEVK_H_
