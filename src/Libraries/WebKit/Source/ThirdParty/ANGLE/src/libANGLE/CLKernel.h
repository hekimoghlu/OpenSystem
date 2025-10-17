/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
// CLKernel.h: Defines the cl::Kernel class, which is a function declared in an OpenCL program.

#ifndef LIBANGLE_CLKERNEL_H_
#define LIBANGLE_CLKERNEL_H_

#include "libANGLE/CLObject.h"
#include "libANGLE/renderer/CLKernelImpl.h"

namespace cl
{

class Kernel final : public _cl_kernel, public Object
{
  public:
    // Front end entry functions, only called from OpenCL entry points

    angle::Result setArg(cl_uint argIndex, size_t argSize, const void *argValue);

    angle::Result getInfo(KernelInfo name,
                          size_t valueSize,
                          void *value,
                          size_t *valueSizeRet) const;

    angle::Result getWorkGroupInfo(cl_device_id device,
                                   KernelWorkGroupInfo name,
                                   size_t valueSize,
                                   void *value,
                                   size_t *valueSizeRet) const;

    angle::Result getArgInfo(cl_uint argIndex,
                             KernelArgInfo name,
                             size_t valueSize,
                             void *value,
                             size_t *valueSizeRet) const;

    const std::string &getName() const { return mInfo.functionName; }

    bool areAllArgsSet() const
    {
        return std::all_of(mSetArguments.begin(), mSetArguments.end(),
                           [](KernelArg arg) { return arg.isSet == true; });
    }

    Kernel *clone() const;

  public:
    ~Kernel() override;

    const Program &getProgram() const;
    const rx::CLKernelImpl::Info &getInfo() const;

    template <typename T = rx::CLKernelImpl>
    T &getImpl() const;

  private:
    Kernel(Program &program, const char *name);
    Kernel(Program &program, const rx::CLKernelImpl::CreateFunc &createFunc);

    void initImpl();

    const ProgramPtr mProgram;
    rx::CLKernelImpl::Ptr mImpl;
    rx::CLKernelImpl::Info mInfo;

    std::vector<KernelArg> mSetArguments;

    friend class Object;
    friend class Program;
};

inline const Program &Kernel::getProgram() const
{
    return *mProgram;
}

inline const rx::CLKernelImpl::Info &Kernel::getInfo() const
{
    return mInfo;
}

template <typename T>
inline T &Kernel::getImpl() const
{
    return static_cast<T &>(*mImpl);
}

}  // namespace cl

#endif  // LIBANGLE_CLKERNEL_H_
