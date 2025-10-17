/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 7, 2022.
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
// CLProgramImpl.h: Defines the abstract rx::CLProgramImpl class.

#ifndef LIBANGLE_RENDERER_CLPROGRAMIMPL_H_
#define LIBANGLE_RENDERER_CLPROGRAMIMPL_H_

#include "libANGLE/renderer/CLKernelImpl.h"

namespace rx
{

class CLProgramImpl : angle::NonCopyable
{
  public:
    using Ptr = std::unique_ptr<CLProgramImpl>;

    CLProgramImpl(const cl::Program &program);
    virtual ~CLProgramImpl();

    virtual angle::Result build(const cl::DevicePtrs &devices,
                                const char *options,
                                cl::Program *notify) = 0;

    virtual angle::Result compile(const cl::DevicePtrs &devices,
                                  const char *options,
                                  const cl::ProgramPtrs &inputHeaders,
                                  const char **headerIncludeNames,
                                  cl::Program *notify) = 0;

    virtual angle::Result getInfo(cl::ProgramInfo name,
                                  size_t valueSize,
                                  void *value,
                                  size_t *valueSizeRet) const = 0;

    virtual angle::Result getBuildInfo(const cl::Device &device,
                                       cl::ProgramBuildInfo name,
                                       size_t valueSize,
                                       void *value,
                                       size_t *valueSizeRet) const = 0;

    virtual angle::Result createKernel(const cl::Kernel &kernel,
                                       const char *name,
                                       CLKernelImpl::Ptr *kernelOut) = 0;

    virtual angle::Result createKernels(cl_uint numKernels,
                                        CLKernelImpl::CreateFuncs &createFuncs,
                                        cl_uint *numKernelsRet) = 0;

  protected:
    const cl::Program &mProgram;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_CLPROGRAMIMPL_H_
