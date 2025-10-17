/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// SyncImpl.h: Defines the rx::SyncImpl class.

#ifndef LIBANGLE_RENDERER_FENCESYNCIMPL_H_
#define LIBANGLE_RENDERER_FENCESYNCIMPL_H_

#include "libANGLE/Error.h"

#include "common/angleutils.h"

#include "angle_gl.h"

namespace gl
{
class Context;
}  // namespace gl

namespace rx
{
class SyncImpl : angle::NonCopyable
{
  public:
    SyncImpl() {}
    virtual ~SyncImpl() {}

    virtual void onDestroy(const gl::Context *context) {}

    virtual angle::Result set(const gl::Context *context, GLenum condition, GLbitfield flags) = 0;
    virtual angle::Result clientWait(const gl::Context *context,
                                     GLbitfield flags,
                                     GLuint64 timeout,
                                     GLenum *outResult)                                       = 0;
    virtual angle::Result serverWait(const gl::Context *context,
                                     GLbitfield flags,
                                     GLuint64 timeout)                                        = 0;
    virtual angle::Result getStatus(const gl::Context *context, GLint *outResult)             = 0;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_FENCESYNCIMPL_H_
