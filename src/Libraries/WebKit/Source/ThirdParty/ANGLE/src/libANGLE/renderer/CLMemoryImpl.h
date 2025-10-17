/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 23, 2023.
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
// CLMemoryImpl.h: Defines the abstract rx::CLMemoryImpl class.

#ifndef LIBANGLE_RENDERER_CLMEMORYIMPL_H_
#define LIBANGLE_RENDERER_CLMEMORYIMPL_H_

#include "libANGLE/renderer/cl_types.h"

namespace rx
{

class CLMemoryImpl : angle::NonCopyable
{
  public:
    using Ptr = std::unique_ptr<CLMemoryImpl>;

    CLMemoryImpl(const cl::Memory &memory);
    virtual ~CLMemoryImpl();

    virtual angle::Result createSubBuffer(const cl::Buffer &buffer,
                                          cl::MemFlags flags,
                                          size_t size,
                                          CLMemoryImpl::Ptr *subBufferOut) = 0;

  protected:
    const cl::Memory &mMemory;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_CLMEMORYIMPL_H_
