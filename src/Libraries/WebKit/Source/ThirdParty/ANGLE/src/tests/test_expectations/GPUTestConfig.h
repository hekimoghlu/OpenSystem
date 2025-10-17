/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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

#ifndef TEST_EXPECTATIONS_GPU_TEST_CONFIG_H_
#define TEST_EXPECTATIONS_GPU_TEST_CONFIG_H_

#include <common/bitset_utils.h>

namespace angle
{

struct GPUTestConfig
{
  public:
    enum API
    {
        kAPIUnknown = 0,
        kAPID3D9,
        kAPID3D11,
        kAPIGLDesktop,
        kAPIGLES,
        kAPIVulkan,
        kAPISwiftShader,
        kAPIMetal,
        kAPIWgpu,
    };

    enum Condition
    {
        kConditionNone = 0,
        kConditionWinXP,
        kConditionWinVista,
        kConditionWin7,
        kConditionWin8,
        kConditionWin10,
        kConditionWin,
        kConditionMacLeopard,
        kConditionMacSnowLeopard,
        kConditionMacLion,
        kConditionMacMountainLion,
        kConditionMacMavericks,
        kConditionMacYosemite,
        kConditionMacElCapitan,
        kConditionMacSierra,
        kConditionMacHighSierra,
        kConditionMacMojave,
        kConditionMac,
        kConditionIOS,
        kConditionLinux,
        kConditionAndroid,
        kConditionNVIDIA,
        kConditionAMD,
        kConditionIntel,
        kConditionVMWare,
        kConditionApple,
        kConditionRelease,
        kConditionDebug,
        kConditionD3D9,
        kConditionD3D11,
        kConditionGLDesktop,
        kConditionGLES,
        kConditionVulkan,
        kConditionMetal,
        kConditionWgpu,
        kConditionNexus5X,
        kConditionPixel2OrXL,
        kConditionPixel4OrXL,
        kConditionPixel6,
        kConditionPixel7,
        kConditionFlipN2,
        kConditionMaliG710,
        kConditionGalaxyA23,
        kConditionGalaxyA34,
        kConditionGalaxyA54,
        kConditionGalaxyS22,
        kConditionGalaxyS23,
        kConditionGalaxyS24Exynos,
        kConditionGalaxyS24Qualcomm,
        kConditionFindX6,
        kConditionNVIDIAQuadroP400,
        kConditionNVIDIAGTX1660,
        kConditionPineapple,
        kConditionSwiftShader,
        kConditionPreRotation,
        kConditionPreRotation90,
        kConditionPreRotation180,
        kConditionPreRotation270,
        kConditionNoSan,
        kConditionASan,
        kConditionTSan,
        kConditionUBSan,

        kNumberOfConditions,
    };

    using ConditionArray = angle::BitSet<GPUTestConfig::kNumberOfConditions>;

    GPUTestConfig();
    GPUTestConfig(bool isSwiftShader);
    GPUTestConfig(const API &api, uint32_t preRotation);

    const GPUTestConfig::ConditionArray &getConditions() const;

  protected:
    GPUTestConfig::ConditionArray mConditions;
};

}  // namespace angle

#endif  // TEST_EXPECTATIONS_GPU_TEST_CONFIG_H_
