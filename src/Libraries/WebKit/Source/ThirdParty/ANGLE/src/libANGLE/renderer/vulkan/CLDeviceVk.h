/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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
// CLDeviceVk.h: Defines the class interface for CLDeviceVk, implementing CLDeviceImpl.

#ifndef LIBANGLE_RENDERER_VULKAN_CLDEVICEVK_H_
#define LIBANGLE_RENDERER_VULKAN_CLDEVICEVK_H_

#include "libANGLE/renderer/vulkan/cl_types.h"
#include "libANGLE/renderer/vulkan/vk_renderer.h"

#include "libANGLE/renderer/CLDeviceImpl.h"

#include "spirv-tools/libspirv.h"

namespace rx
{

class CLDeviceVk : public CLDeviceImpl
{
  public:
    explicit CLDeviceVk(const cl::Device &device, vk::Renderer *renderer);
    ~CLDeviceVk() override;

    Info createInfo(cl::DeviceType type) const override;

    const vk::Renderer *getRenderer() const { return mRenderer; }
    const cl::Device &getFrontendObject() const { return const_cast<cl::Device &>(mDevice); }
    angle::Result getInfoUInt(cl::DeviceInfo name, cl_uint *value) const override;
    angle::Result getInfoULong(cl::DeviceInfo name, cl_ulong *value) const override;
    angle::Result getInfoSizeT(cl::DeviceInfo name, size_t *value) const override;
    angle::Result getInfoStringLength(cl::DeviceInfo name, size_t *value) const override;
    angle::Result getInfoString(cl::DeviceInfo name, size_t size, char *value) const override;

    angle::Result createSubDevices(const cl_device_partition_property *properties,
                                   cl_uint numDevices,
                                   CreateFuncs &subDevices,
                                   cl_uint *numDevicesRet) override;

    // Returns runtime-selected LWS value
    cl::WorkgroupSize selectWorkGroupSize(const cl::NDRange &ndrange) const;

    spv_target_env getSpirvVersion() const { return mSpirvVersion; }

  private:
    vk::Renderer *mRenderer;
    spv_target_env mSpirvVersion;
    angle::HashMap<cl::DeviceInfo, cl_uint> mInfoUInt;
    angle::HashMap<cl::DeviceInfo, cl_ulong> mInfoULong;
    angle::HashMap<cl::DeviceInfo, size_t> mInfoSizeT;
    angle::HashMap<cl::DeviceInfo, std::string> mInfoString;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_CLDEVICEVK_H_
