/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 1, 2024.
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
// CLEvent.h: Defines the cl::Event class, which can be used to track the execution status of an
// OpenCL command.

#ifndef LIBANGLE_CLEVENT_H_
#define LIBANGLE_CLEVENT_H_

#include "libANGLE/CLObject.h"
#include "libANGLE/renderer/CLEventImpl.h"

#include "common/SynchronizedValue.h"

#include <array>

namespace cl
{

class Event final : public _cl_event, public Object
{
  public:
    // Front end entry functions, only called from OpenCL entry points

    angle::Result setUserEventStatus(cl_int executionStatus);

    angle::Result getInfo(EventInfo name,
                          size_t valueSize,
                          void *value,
                          size_t *valueSizeRet) const;

    angle::Result setCallback(cl_int commandExecCallbackType, EventCB pfnNotify, void *userData);

    angle::Result getProfilingInfo(ProfilingInfo name,
                                   size_t valueSize,
                                   void *value,
                                   size_t *valueSizeRet);

    bool isUserEvent() const { return mCommandType == CL_COMMAND_USER; }

  public:
    ~Event() override;

    Context &getContext();
    const Context &getContext() const;
    const CommandQueuePtr &getCommandQueue() const;
    cl_command_type getCommandType() const;
    bool wasStatusChanged() const;

    template <typename T = rx::CLEventImpl>
    T &getImpl() const;

    void callback(cl_int commandStatus);

    static EventPtrs Cast(cl_uint numEvents, const cl_event *eventList);

  private:
    using CallbackData = std::pair<EventCB, void *>;
    using Callbacks    = std::vector<CallbackData>;

    Event(Context &context);

    Event(CommandQueue &queue,
          cl_command_type commandType,
          const rx::CLEventImpl::CreateFunc &createFunc);

    const ContextPtr mContext;
    const CommandQueuePtr mCommandQueue;
    const cl_command_type mCommandType;
    rx::CLEventImpl::Ptr mImpl;

    bool mStatusWasChanged = false;

    // Create separate storage for each possible callback type.
    static_assert(CL_COMPLETE == 0 && CL_RUNNING == 1 && CL_SUBMITTED == 2,
                  "OpenCL command execution status values are not as assumed");
    angle::SynchronizedValue<std::array<Callbacks, 3u>> mCallbacks;

    friend class Object;
};

inline Context &Event::getContext()
{
    return *mContext;
}

inline const Context &Event::getContext() const
{
    return *mContext;
}

inline const CommandQueuePtr &Event::getCommandQueue() const
{
    return mCommandQueue;
}

inline cl_command_type Event::getCommandType() const
{
    return mCommandType;
}

inline bool Event::wasStatusChanged() const
{
    return mStatusWasChanged;
}

template <typename T>
inline T &Event::getImpl() const
{
    return static_cast<T &>(*mImpl);
}

}  // namespace cl

#endif  // LIBANGLE_CLEVENT_H_
