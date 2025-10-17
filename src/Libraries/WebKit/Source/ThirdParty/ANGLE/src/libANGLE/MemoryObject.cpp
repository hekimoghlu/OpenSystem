/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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
// MemoryObject.h: Implements the gl::MemoryObject class [EXT_external_objects]

#include "libANGLE/MemoryObject.h"

#include "common/angleutils.h"
#include "libANGLE/renderer/GLImplFactory.h"
#include "libANGLE/renderer/MemoryObjectImpl.h"

namespace gl
{

MemoryObject::MemoryObject(rx::GLImplFactory *factory, MemoryObjectID id)
    : RefCountObject(factory->generateSerial(), id),
      mImplementation(factory->createMemoryObject()),
      mImmutable(false),
      mDedicatedMemory(false),
      mProtectedMemory(false)
{}

MemoryObject::~MemoryObject() {}

void MemoryObject::onDestroy(const Context *context)
{
    mImplementation->onDestroy(context);
}

angle::Result MemoryObject::setDedicatedMemory(const Context *context, bool dedicatedMemory)
{
    ANGLE_TRY(mImplementation->setDedicatedMemory(context, dedicatedMemory));
    mDedicatedMemory = dedicatedMemory;
    return angle::Result::Continue;
}

angle::Result MemoryObject::setProtectedMemory(const Context *context, bool protectedMemory)
{
    ANGLE_TRY(mImplementation->setProtectedMemory(context, protectedMemory));
    mProtectedMemory = protectedMemory;
    return angle::Result::Continue;
}

angle::Result MemoryObject::importFd(Context *context,
                                     GLuint64 size,
                                     HandleType handleType,
                                     GLint fd)
{
    ANGLE_TRY(mImplementation->importFd(context, size, handleType, fd));
    mImmutable = true;
    return angle::Result::Continue;
}

angle::Result MemoryObject::importZirconHandle(Context *context,
                                               GLuint64 size,
                                               HandleType handleType,
                                               GLuint handle)
{
    ANGLE_TRY(mImplementation->importZirconHandle(context, size, handleType, handle));
    mImmutable = true;
    return angle::Result::Continue;
}

}  // namespace gl
