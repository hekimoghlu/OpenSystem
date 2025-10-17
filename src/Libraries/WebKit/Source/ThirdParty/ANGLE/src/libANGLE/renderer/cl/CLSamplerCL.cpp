/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
// CLSamplerCL.cpp: Implements the class methods for CLSamplerCL.

#include "libANGLE/renderer/cl/CLSamplerCL.h"

#include "libANGLE/renderer/cl/CLContextCL.h"

#include "libANGLE/CLContext.h"
#include "libANGLE/CLSampler.h"

namespace rx
{

CLSamplerCL::CLSamplerCL(const cl::Sampler &sampler, cl_sampler native)
    : CLSamplerImpl(sampler), mNative(native)
{
    sampler.getContext().getImpl<CLContextCL>().mData->mSamplers.emplace(sampler.getNative());
}

CLSamplerCL::~CLSamplerCL()
{
    const size_t numRemoved =
        mSampler.getContext().getImpl<CLContextCL>().mData->mSamplers.erase(mSampler.getNative());
    ASSERT(numRemoved == 1u);

    if (mNative->getDispatch().clReleaseSampler(mNative) != CL_SUCCESS)
    {
        ERR() << "Error while releasing CL sampler";
    }
}

}  // namespace rx
