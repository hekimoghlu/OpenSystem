/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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
// CLEventVk.h: Defines the class interface for CLEventVk, implementing CLEventImpl.

#ifndef LIBANGLE_RENDERER_VULKAN_CLEVENTVK_H_
#define LIBANGLE_RENDERER_VULKAN_CLEVENTVK_H_

#include <condition_variable>
#include <mutex>

#include "libANGLE/renderer/vulkan/CLContextVk.h"
#include "libANGLE/renderer/vulkan/cl_types.h"
#include "libANGLE/renderer/vulkan/vk_resource.h"

#include "libANGLE/renderer/CLEventImpl.h"

#include "libANGLE/CLCommandQueue.h"
#include "libANGLE/CLContext.h"
#include "libANGLE/CLEvent.h"

namespace rx
{

class CLEventVk : public CLEventImpl, public vk::Resource
{
  public:
    CLEventVk(const cl::Event &event);
    ~CLEventVk() override;

    cl_int getCommandType() const { return mEvent.getCommandType(); }
    bool isUserEvent() const { return mEvent.isUserEvent(); }
    cl::Event &getFrontendObject() { return const_cast<cl::Event &>(mEvent); }

    angle::Result getCommandExecutionStatus(cl_int &executionStatus) override;

    angle::Result setUserEventStatus(cl_int executionStatus) override;

    angle::Result setCallback(cl::Event &event, cl_int commandExecCallbackType) override;

    angle::Result getProfilingInfo(cl::ProfilingInfo name,
                                   size_t valueSize,
                                   void *value,
                                   size_t *valueSizeRet) override;

    angle::Result waitForUserEventStatus();
    angle::Result setStatusAndExecuteCallback(cl_int status);
    angle::Result setTimestamp(cl_int status);

  private:
    std::mutex mUserEventMutex;
    angle::SynchronizedValue<cl_int> mStatus;
    std::condition_variable mUserEventCondition;
    angle::SynchronizedValue<cl::EventStatusMap<bool>> mHaveCallbacks;

    // Event profiling timestamps
    struct ProfilingTimestamps
    {
        cl_ulong commandEndTS;
        cl_ulong commandStartTS;
        cl_ulong commandQueuedTS;
        cl_ulong commandSubmitTS;
        cl_ulong commandCompleteTS;
    };
    angle::SynchronizedValue<ProfilingTimestamps> mProfilingTimestamps;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_CLEVENTVK_H_
