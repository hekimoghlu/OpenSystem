/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
// CLDeviceCL.h: Defines the class interface for CLDeviceCL, implementing CLDeviceImpl.

#ifndef LIBANGLE_RENDERER_CL_CLDEVICECL_H_
#define LIBANGLE_RENDERER_CL_CLDEVICECL_H_

#include "libANGLE/renderer/CLDeviceImpl.h"

namespace rx
{

class CLDeviceCL : public CLDeviceImpl
{
  public:
    ~CLDeviceCL() override;

    cl_device_id getNative() const;

    Info createInfo(cl::DeviceType type) const override;

    angle::Result getInfoUInt(cl::DeviceInfo name, cl_uint *value) const override;
    angle::Result getInfoULong(cl::DeviceInfo name, cl_ulong *value) const override;
    angle::Result getInfoSizeT(cl::DeviceInfo name, size_t *value) const override;
    angle::Result getInfoStringLength(cl::DeviceInfo name, size_t *value) const override;
    angle::Result getInfoString(cl::DeviceInfo name, size_t size, char *value) const override;

    angle::Result createSubDevices(const cl_device_partition_property *properties,
                                   cl_uint numDevices,
                                   CreateFuncs &createFuncs,
                                   cl_uint *numDevicesRet) override;

  private:
    CLDeviceCL(const cl::Device &device, cl_device_id native);

    const cl_device_id mNative;

    friend class CLPlatformCL;
};

inline cl_device_id CLDeviceCL::getNative() const
{
    return mNative;
}

}  // namespace rx

#endif  // LIBANGLE_RENDERER_CL_CLDEVICECL_H_
