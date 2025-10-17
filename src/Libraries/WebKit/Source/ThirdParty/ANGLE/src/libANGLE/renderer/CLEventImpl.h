/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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
// CLEventImpl.h: Defines the abstract rx::CLEventImpl class.

#ifndef LIBANGLE_RENDERER_CLEVENTIMPL_H_
#define LIBANGLE_RENDERER_CLEVENTIMPL_H_

#include "libANGLE/renderer/cl_types.h"

namespace rx
{

class CLEventImpl : angle::NonCopyable
{
  public:
    using Ptr        = std::unique_ptr<CLEventImpl>;
    using CreateFunc = std::function<Ptr(const cl::Event &)>;

    CLEventImpl(const cl::Event &event);
    virtual ~CLEventImpl();

    virtual angle::Result getCommandExecutionStatus(cl_int &executionStatus) = 0;

    virtual angle::Result setUserEventStatus(cl_int executionStatus) = 0;

    virtual angle::Result setCallback(cl::Event &event, cl_int commandExecCallbackType) = 0;

    virtual angle::Result getProfilingInfo(cl::ProfilingInfo name,
                                           size_t valueSize,
                                           void *value,
                                           size_t *valueSizeRet) = 0;

  protected:
    const cl::Event &mEvent;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_CLEVENTIMPL_H_
