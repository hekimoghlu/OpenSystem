/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// vulkan_icd.h : Helper for creating vulkan instances & selecting physical device.

#ifndef COMMON_VULKAN_VULKAN_ICD_H_
#define COMMON_VULKAN_VULKAN_ICD_H_

#include <string>

#include "common/Optional.h"
#include "common/angleutils.h"
#include "common/vulkan/vk_headers.h"

namespace angle
{

namespace vk
{

// The minimum version of Vulkan that ANGLE requires.  If an instance or device below this version
// is encountered, initialization will skip the device if possible, or if no other suitable device
// is available then initialization will fail.
constexpr uint32_t kMinimumVulkanAPIVersion = VK_API_VERSION_1_1;

enum class ICD
{
    Default,
    Mock,
    SwiftShader,
};

struct SimpleDisplayWindow
{
    uint16_t width;
    uint16_t height;
};

class [[nodiscard]] ScopedVkLoaderEnvironment : angle::NonCopyable
{
  public:
    ScopedVkLoaderEnvironment(bool enableDebugLayers, vk::ICD icd);
    ~ScopedVkLoaderEnvironment();

    bool canEnableDebugLayers() const { return mEnableDebugLayers; }
    vk::ICD getEnabledICD() const { return mICD; }

  private:
    bool setICDEnvironment(const char *icd);

    bool mEnableDebugLayers;
    vk::ICD mICD;
    bool mChangedCWD;
    Optional<std::string> mPreviousCWD;
    bool mChangedICDEnv;
    Optional<std::string> mPreviousICDEnv;
    Optional<std::string> mPreviousCustomExtensionsEnv;
    bool mChangedNoDeviceSelect;
    Optional<std::string> mPreviousNoDeviceSelectEnv;
};

void ChoosePhysicalDevice(PFN_vkGetPhysicalDeviceProperties2 pGetPhysicalDeviceProperties2,
                          const std::vector<VkPhysicalDevice> &physicalDevices,
                          vk::ICD preferredICD,
                          uint32_t preferredVendorID,
                          uint32_t preferredDeviceID,
                          const uint8_t *preferredDeviceUUID,
                          const uint8_t *preferredDriverUUID,
                          VkDriverId preferredDriverID,
                          VkPhysicalDevice *physicalDeviceOut,
                          VkPhysicalDeviceProperties2 *physicalDeviceProperties2Out,
                          VkPhysicalDeviceIDProperties *physicalDeviceIDPropertiesOut,
                          VkPhysicalDeviceDriverProperties *physicalDeviceDriverPropertiesOut);

}  // namespace vk

}  // namespace angle

#endif  // COMMON_VULKAN_VULKAN_ICD_H_
