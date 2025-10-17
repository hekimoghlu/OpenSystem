/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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

// SystemInfo_ios.cpp: implementation of the iOS-specific parts of SystemInfo.h

#include "gpu_info_util/SystemInfo_internal.h"

namespace angle
{

bool GetSystemInfo_ios(SystemInfo *info)
{
    {
        // TODO(anglebug.com/42262902): Get the actual system version and GPU info.
        info->machineModelVersion = "0.0";
        GPUDeviceInfo deviceInfo;
        deviceInfo.vendorId = kVendorID_Apple;
        info->gpus.push_back(deviceInfo);
    }

    return true;
}

}  // namespace angle
