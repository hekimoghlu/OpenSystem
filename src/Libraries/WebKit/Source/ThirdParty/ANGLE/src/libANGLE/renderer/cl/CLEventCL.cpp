/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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
// CLEventCL.cpp: Implements the class methods for CLEventCL.

#include "libANGLE/renderer/cl/CLEventCL.h"

#include "libANGLE/CLEvent.h"
#include "libANGLE/cl_utils.h"

namespace rx
{

CLEventCL::CLEventCL(const cl::Event &event, cl_event native) : CLEventImpl(event), mNative(native)
{}

CLEventCL::~CLEventCL()
{
    if (mNative->getDispatch().clReleaseEvent(mNative) != CL_SUCCESS)
    {
        ERR() << "Error while releasing CL event";
    }
}

angle::Result CLEventCL::getCommandExecutionStatus(cl_int &executionStatus)
{
    ANGLE_CL_TRY(mNative->getDispatch().clGetEventInfo(mNative, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                                       sizeof(executionStatus), &executionStatus,
                                                       nullptr));
    return angle::Result::Continue;
}

angle::Result CLEventCL::setUserEventStatus(cl_int executionStatus)
{
    ANGLE_CL_TRY(mNative->getDispatch().clSetUserEventStatus(mNative, executionStatus));
    return angle::Result::Continue;
}

angle::Result CLEventCL::setCallback(cl::Event &event, cl_int commandExecCallbackType)
{
    ANGLE_CL_TRY(mNative->getDispatch().clSetEventCallback(mNative, commandExecCallbackType,
                                                           Callback, &event));
    return angle::Result::Continue;
}

angle::Result CLEventCL::getProfilingInfo(cl::ProfilingInfo name,
                                          size_t valueSize,
                                          void *value,
                                          size_t *valueSizeRet)
{
    ANGLE_CL_TRY(mNative->getDispatch().clGetEventProfilingInfo(mNative, cl::ToCLenum(name),
                                                                valueSize, value, valueSizeRet));
    return angle::Result::Continue;
}

std::vector<cl_event> CLEventCL::Cast(const cl::EventPtrs &events)
{
    std::vector<cl_event> nativeEvents;
    nativeEvents.reserve(events.size());
    for (const cl::EventPtr &event : events)
    {
        nativeEvents.emplace_back(event->getImpl<CLEventCL>().getNative());
    }
    return nativeEvents;
}

void CLEventCL::Callback(cl_event event, cl_int commandStatus, void *userData)
{
    static_cast<cl::Event *>(userData)->callback(commandStatus);
}

}  // namespace rx
