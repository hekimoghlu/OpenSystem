/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RenderbufferNULL.cpp:
//    Implements the class methods for RenderbufferNULL.
//

#include "libANGLE/renderer/null/RenderbufferNULL.h"

#include "common/debug.h"

namespace rx
{

RenderbufferNULL::RenderbufferNULL(const gl::RenderbufferState &state) : RenderbufferImpl(state) {}

RenderbufferNULL::~RenderbufferNULL() {}

angle::Result RenderbufferNULL::setStorage(const gl::Context *context,
                                           GLenum internalformat,
                                           GLsizei width,
                                           GLsizei height)
{
    return angle::Result::Continue;
}

angle::Result RenderbufferNULL::setStorageMultisample(const gl::Context *context,
                                                      GLsizei samples,
                                                      GLenum internalformat,
                                                      GLsizei width,
                                                      GLsizei height,
                                                      gl::MultisamplingMode mode)
{
    return angle::Result::Continue;
}

angle::Result RenderbufferNULL::setStorageEGLImageTarget(const gl::Context *context,
                                                         egl::Image *image)
{
    return angle::Result::Continue;
}

angle::Result RenderbufferNULL::initializeContents(const gl::Context *context,
                                                   GLenum binding,
                                                   const gl::ImageIndex &imageIndex)
{
    return angle::Result::Continue;
}

}  // namespace rx
