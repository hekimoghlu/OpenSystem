/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
// CLSampler.cpp: Implements the cl::Sampler class.

#include "libANGLE/CLSampler.h"

#include "libANGLE/CLContext.h"
#include "libANGLE/cl_utils.h"

#include <cstring>

namespace cl
{

angle::Result Sampler::getInfo(SamplerInfo name,
                               size_t valueSize,
                               void *value,
                               size_t *valueSizeRet) const
{
    static_assert(std::is_same<cl_uint, cl_addressing_mode>::value &&
                      std::is_same<cl_uint, cl_filter_mode>::value,
                  "OpenCL type mismatch");

    cl_uint valUInt       = 0u;
    void *valPointer      = nullptr;
    const void *copyValue = nullptr;
    size_t copySize       = 0u;

    switch (name)
    {
        case SamplerInfo::ReferenceCount:
            valUInt   = getRefCount();
            copyValue = &valUInt;
            copySize  = sizeof(valUInt);
            break;
        case SamplerInfo::Context:
            valPointer = mContext->getNative();
            copyValue  = &valPointer;
            copySize   = sizeof(valPointer);
            break;
        case SamplerInfo::NormalizedCoords:
            copyValue = &mNormalizedCoords;
            copySize  = sizeof(mNormalizedCoords);
            break;
        case SamplerInfo::AddressingMode:
            valUInt   = ToCLenum(mAddressingMode);
            copyValue = &valUInt;
            copySize  = sizeof(valUInt);
            break;
        case SamplerInfo::FilterMode:
            valUInt   = ToCLenum(mFilterMode);
            copyValue = &valUInt;
            copySize  = sizeof(valUInt);
            break;
        case SamplerInfo::Properties:
            copyValue = mProperties.data();
            copySize  = mProperties.size() * sizeof(decltype(mProperties)::value_type);
            break;
        default:
            ANGLE_CL_RETURN_ERROR(CL_INVALID_VALUE);
    }

    if (value != nullptr)
    {
        // CL_INVALID_VALUE if size in bytes specified by param_value_size is < size of return type
        // as described in the Sampler Object Queries table and param_value is not NULL.
        if (valueSize < copySize)
        {
            ANGLE_CL_RETURN_ERROR(CL_INVALID_VALUE);
        }
        if (copyValue != nullptr)
        {
            std::memcpy(value, copyValue, copySize);
        }
    }
    if (valueSizeRet != nullptr)
    {
        *valueSizeRet = copySize;
    }
    return angle::Result::Continue;
}

Sampler::~Sampler() = default;

Sampler::Sampler(Context &context,
                 PropArray &&properties,
                 cl_bool normalizedCoords,
                 AddressingMode addressingMode,
                 FilterMode filterMode)
    : mContext(&context),
      mProperties(std::move(properties)),
      mNormalizedCoords(normalizedCoords),
      mAddressingMode(addressingMode),
      mFilterMode(filterMode),
      mImpl(nullptr)
{
    ANGLE_CL_IMPL_TRY(context.getImpl().createSampler(*this, &mImpl));
}

}  // namespace cl
