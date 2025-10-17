/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
// CLKernelCL.h: Defines the class interface for CLKernelCL, implementing CLKernelImpl.

#ifndef LIBANGLE_RENDERER_CL_CLKERNELCL_H_
#define LIBANGLE_RENDERER_CL_CLKERNELCL_H_

#include "libANGLE/renderer/cl/cl_types.h"

#include "libANGLE/renderer/CLKernelImpl.h"

namespace rx
{

class CLKernelCL : public CLKernelImpl
{
  public:
    CLKernelCL(const cl::Kernel &kernel, cl_kernel native);
    ~CLKernelCL() override;

    cl_kernel getNative() const;

    angle::Result setArg(cl_uint argIndex, size_t argSize, const void *argValue) override;

    angle::Result createInfo(Info *infoOut) const override;

  private:
    const cl_kernel mNative;
};

inline cl_kernel CLKernelCL::getNative() const
{
    return mNative;
}

}  // namespace rx

#endif  // LIBANGLE_RENDERER_CL_CLKERNELCL_H_
