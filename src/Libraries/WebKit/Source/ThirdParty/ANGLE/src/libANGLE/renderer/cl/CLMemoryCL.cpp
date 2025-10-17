/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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
// CLMemoryCL.cpp: Implements the class methods for CLMemoryCL.

#include "libANGLE/renderer/cl/CLMemoryCL.h"

#include "libANGLE/renderer/cl/CLContextCL.h"

#include "libANGLE/CLBuffer.h"
#include "libANGLE/CLContext.h"
#include "libANGLE/cl_utils.h"

namespace rx
{

CLMemoryCL::CLMemoryCL(const cl::Memory &memory, cl_mem native)
    : CLMemoryImpl(memory), mNative(native)
{
    memory.getContext().getImpl<CLContextCL>().mData->mMemories.emplace(memory.getNative());
}

CLMemoryCL::~CLMemoryCL()
{
    const size_t numRemoved =
        mMemory.getContext().getImpl<CLContextCL>().mData->mMemories.erase(mMemory.getNative());
    ASSERT(numRemoved == 1u);

    if (mNative->getDispatch().clReleaseMemObject(mNative) != CL_SUCCESS)
    {
        ERR() << "Error while releasing CL memory object";
    }
}

angle::Result CLMemoryCL::createSubBuffer(const cl::Buffer &buffer,
                                          cl::MemFlags flags,
                                          size_t size,
                                          CLMemoryImpl::Ptr *subBufferOut)
{
    cl_int errorCode              = CL_SUCCESS;
    const cl_buffer_region region = {buffer.getOffset(), size};

    const cl_mem nativeBuffer = mNative->getDispatch().clCreateSubBuffer(
        mNative, flags.get(), CL_BUFFER_CREATE_TYPE_REGION, &region, &errorCode);
    ANGLE_CL_TRY(errorCode);

    *subBufferOut =
        CLMemoryImpl::Ptr(nativeBuffer != nullptr ? new CLMemoryCL(buffer, nativeBuffer) : nullptr);
    return angle::Result::Continue;
}

}  // namespace rx
