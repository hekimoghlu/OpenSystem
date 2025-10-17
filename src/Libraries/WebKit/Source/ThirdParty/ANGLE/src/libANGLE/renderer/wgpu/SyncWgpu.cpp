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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SyncWgpu.cpp:
//    Implements the class methods for SyncWgpu.
//

#include "libANGLE/renderer/wgpu/SyncWgpu.h"

#include "common/debug.h"

namespace rx
{

SyncWgpu::SyncWgpu() : SyncImpl() {}

SyncWgpu::~SyncWgpu() {}

angle::Result SyncWgpu::set(const gl::Context *context, GLenum condition, GLbitfield flags)
{
    return angle::Result::Continue;
}

angle::Result SyncWgpu::clientWait(const gl::Context *context,
                                   GLbitfield flags,
                                   GLuint64 timeout,
                                   GLenum *outResult)
{
    *outResult = GL_ALREADY_SIGNALED;
    return angle::Result::Continue;
}

angle::Result SyncWgpu::serverWait(const gl::Context *context, GLbitfield flags, GLuint64 timeout)
{
    return angle::Result::Continue;
}

angle::Result SyncWgpu::getStatus(const gl::Context *context, GLint *outResult)
{
    *outResult = GL_SIGNALED;
    return angle::Result::Continue;
}

}  // namespace rx
