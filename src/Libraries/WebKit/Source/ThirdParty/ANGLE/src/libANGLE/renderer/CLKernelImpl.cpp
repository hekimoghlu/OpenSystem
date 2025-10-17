/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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
// CLKernelImpl.cpp: Implements the class methods for CLKernelImpl.

#include "libANGLE/renderer/CLKernelImpl.h"

namespace rx
{

CLKernelImpl::WorkGroupInfo::WorkGroupInfo() = default;

CLKernelImpl::WorkGroupInfo::~WorkGroupInfo() = default;

CLKernelImpl::WorkGroupInfo::WorkGroupInfo(WorkGroupInfo &&) = default;

CLKernelImpl::WorkGroupInfo &CLKernelImpl::WorkGroupInfo::operator=(WorkGroupInfo &&) = default;

CLKernelImpl::ArgInfo::ArgInfo() = default;

CLKernelImpl::ArgInfo::~ArgInfo() = default;

CLKernelImpl::ArgInfo::ArgInfo(ArgInfo &&) = default;

CLKernelImpl::ArgInfo &CLKernelImpl::ArgInfo::operator=(ArgInfo &&) = default;

CLKernelImpl::Info::Info() = default;

CLKernelImpl::Info::~Info() = default;

CLKernelImpl::Info::Info(Info &&) = default;

CLKernelImpl::Info &CLKernelImpl::Info::operator=(Info &&) = default;

CLKernelImpl::CLKernelImpl(const cl::Kernel &kernel) : mKernel(kernel) {}

CLKernelImpl::~CLKernelImpl() = default;

}  // namespace rx
