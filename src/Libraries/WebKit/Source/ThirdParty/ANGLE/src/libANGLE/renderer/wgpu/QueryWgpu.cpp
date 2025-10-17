/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
// QueryWgpu.cpp:
//    Implements the class methods for QueryWgpu.
//

#include "libANGLE/renderer/wgpu/QueryWgpu.h"

#include "common/debug.h"

namespace rx
{

QueryWgpu::QueryWgpu(gl::QueryType type) : QueryImpl(type) {}

QueryWgpu::~QueryWgpu() {}

angle::Result QueryWgpu::begin(const gl::Context *context)
{
    return angle::Result::Continue;
}

angle::Result QueryWgpu::end(const gl::Context *context)
{
    return angle::Result::Continue;
}

angle::Result QueryWgpu::queryCounter(const gl::Context *context)
{
    return angle::Result::Continue;
}

angle::Result QueryWgpu::getResult(const gl::Context *context, GLint *params)
{
    *params = 0;
    return angle::Result::Continue;
}

angle::Result QueryWgpu::getResult(const gl::Context *context, GLuint *params)
{
    *params = 0;
    return angle::Result::Continue;
}

angle::Result QueryWgpu::getResult(const gl::Context *context, GLint64 *params)
{
    *params = 0;
    return angle::Result::Continue;
}

angle::Result QueryWgpu::getResult(const gl::Context *context, GLuint64 *params)
{
    *params = 0;
    return angle::Result::Continue;
}

angle::Result QueryWgpu::isResultAvailable(const gl::Context *context, bool *available)
{
    *available = true;
    return angle::Result::Continue;
}

}  // namespace rx
