/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// MemoryObjectImpl.h: Implements the rx::MemoryObjectImpl class [EXT_external_objects]

#ifndef LIBANGLE_RENDERER_MEMORYOBJECTIMPL_H_
#define LIBANGLE_RENDERER_MEMORYOBJECTIMPL_H_

#include "angle_gl.h"
#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"

namespace gl
{
class Context;
class MemoryObject;
}  // namespace gl

namespace rx
{

class MemoryObjectImpl : angle::NonCopyable
{
  public:
    virtual ~MemoryObjectImpl() {}

    virtual void onDestroy(const gl::Context *context) = 0;

    virtual angle::Result setDedicatedMemory(const gl::Context *context, bool dedicatedMemory) = 0;
    virtual angle::Result setProtectedMemory(const gl::Context *context, bool protectedMemory) = 0;

    virtual angle::Result importFd(gl::Context *context,
                                   GLuint64 size,
                                   gl::HandleType handleType,
                                   GLint fd)                = 0;
    virtual angle::Result importZirconHandle(gl::Context *context,
                                             GLuint64 size,
                                             gl::HandleType handleType,
                                             GLuint handle) = 0;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_MEMORYOBJECTIMPL_H_
