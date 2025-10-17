/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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

#include "libANGLE/renderer/gl/SemaphoreGL.h"

#include "libANGLE/Context.h"
#include "libANGLE/renderer/gl/BufferGL.h"
#include "libANGLE/renderer/gl/ContextGL.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"
#include "libANGLE/renderer/gl/TextureGL.h"
#include "libANGLE/renderer/gl/renderergl_utils.h"

namespace rx
{
namespace
{
void GatherNativeBufferIDs(const gl::BufferBarrierVector &bufferBarriers,
                           gl::BarrierVector<GLuint> *outIDs)
{
    outIDs->resize(bufferBarriers.size());
    for (GLuint bufferIdx = 0; bufferIdx < bufferBarriers.size(); bufferIdx++)
    {
        (*outIDs)[bufferIdx] = GetImplAs<BufferGL>(bufferBarriers[bufferIdx])->getBufferID();
    }
}

void GatherNativeTextureIDs(const gl::TextureBarrierVector &textureBarriers,
                            gl::BarrierVector<GLuint> *outIDs,
                            gl::BarrierVector<GLenum> *outLayouts)
{
    outIDs->resize(textureBarriers.size());
    outLayouts->resize(textureBarriers.size());
    for (GLuint textureIdx = 0; textureIdx < textureBarriers.size(); textureIdx++)
    {
        (*outIDs)[textureIdx] =
            GetImplAs<TextureGL>(textureBarriers[textureIdx].texture)->getTextureID();
        (*outLayouts)[textureIdx] = textureBarriers[textureIdx].layout;
    }
}

}  // anonymous namespace

SemaphoreGL::SemaphoreGL(GLuint semaphoreID) : mSemaphoreID(semaphoreID)
{
    ASSERT(mSemaphoreID != 0);
}

SemaphoreGL::~SemaphoreGL()
{
    ASSERT(mSemaphoreID == 0);
}

void SemaphoreGL::onDestroy(const gl::Context *context)
{
    const FunctionsGL *functions = GetFunctionsGL(context);
    functions->deleteSemaphoresEXT(1, &mSemaphoreID);
    mSemaphoreID = 0;
}

angle::Result SemaphoreGL::importFd(gl::Context *context, gl::HandleType handleType, GLint fd)
{
    const FunctionsGL *functions = GetFunctionsGL(context);
    functions->importSemaphoreFdEXT(mSemaphoreID, ToGLenum(handleType), fd);
    return angle::Result::Continue;
}

angle::Result SemaphoreGL::importZirconHandle(gl::Context *context,
                                              gl::HandleType handleType,
                                              GLuint handle)
{
    UNREACHABLE();
    return angle::Result::Stop;
}

angle::Result SemaphoreGL::wait(gl::Context *context,
                                const gl::BufferBarrierVector &bufferBarriers,
                                const gl::TextureBarrierVector &textureBarriers)
{
    const FunctionsGL *functions = GetFunctionsGL(context);

    gl::BarrierVector<GLuint> bufferIDs(bufferBarriers.size());
    GatherNativeBufferIDs(bufferBarriers, &bufferIDs);

    gl::BarrierVector<GLuint> textureIDs(textureBarriers.size());
    gl::BarrierVector<GLenum> textureLayouts(textureBarriers.size());
    GatherNativeTextureIDs(textureBarriers, &textureIDs, &textureLayouts);
    ASSERT(textureIDs.size() == textureLayouts.size());

    functions->waitSemaphoreEXT(mSemaphoreID, static_cast<GLuint>(bufferIDs.size()),
                                bufferIDs.data(), static_cast<GLuint>(textureIDs.size()),
                                textureIDs.data(), textureLayouts.data());

    return angle::Result::Continue;
}

angle::Result SemaphoreGL::signal(gl::Context *context,
                                  const gl::BufferBarrierVector &bufferBarriers,
                                  const gl::TextureBarrierVector &textureBarriers)
{
    const FunctionsGL *functions = GetFunctionsGL(context);

    gl::BarrierVector<GLuint> bufferIDs(bufferBarriers.size());
    GatherNativeBufferIDs(bufferBarriers, &bufferIDs);

    gl::BarrierVector<GLuint> textureIDs(textureBarriers.size());
    gl::BarrierVector<GLenum> textureLayouts(textureBarriers.size());
    GatherNativeTextureIDs(textureBarriers, &textureIDs, &textureLayouts);
    ASSERT(textureIDs.size() == textureLayouts.size());

    functions->signalSemaphoreEXT(mSemaphoreID, static_cast<GLuint>(bufferIDs.size()),
                                  bufferIDs.data(), static_cast<GLuint>(textureIDs.size()),
                                  textureIDs.data(), textureLayouts.data());

    return angle::Result::Continue;
}

GLuint SemaphoreGL::getSemaphoreID() const
{
    return mSemaphoreID;
}
}  // namespace rx
