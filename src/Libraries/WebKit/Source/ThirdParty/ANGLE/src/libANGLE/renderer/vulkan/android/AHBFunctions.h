/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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

#ifndef LIBANGLE_RENDERER_VULKAN_ANDROID_AHBFUNCTIONS_H_
#define LIBANGLE_RENDERER_VULKAN_ANDROID_AHBFUNCTIONS_H_

#include <android/hardware_buffer.h>

namespace rx
{

class AHBFunctions
{
  public:
    AHBFunctions();
    ~AHBFunctions();

    void acquire(AHardwareBuffer *buffer) const { mAcquireFn(buffer); }
    void describe(const AHardwareBuffer *buffer, AHardwareBuffer_Desc *outDesc) const
    {
        mDescribeFn(buffer, outDesc);
    }
    void release(AHardwareBuffer *buffer) const { mReleaseFn(buffer); }

    bool valid() const { return mAcquireFn && mDescribeFn && mReleaseFn; }

  private:
    using PFN_AHARDWAREBUFFER_acquire  = void (*)(AHardwareBuffer *buffer);
    using PFN_AHARDWAREBUFFER_describe = void (*)(const AHardwareBuffer *buffer,
                                                  AHardwareBuffer_Desc *outDesc);
    using PFN_AHARDWAREBUFFER_release  = void (*)(AHardwareBuffer *buffer);

    PFN_AHARDWAREBUFFER_acquire mAcquireFn   = nullptr;
    PFN_AHARDWAREBUFFER_describe mDescribeFn = nullptr;
    PFN_AHARDWAREBUFFER_release mReleaseFn   = nullptr;

    void getAhbProcAddresses(void *handle);

    void *mLibNativeWindowHandle;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_ANDROID_AHBFUNCTIONS_H_
