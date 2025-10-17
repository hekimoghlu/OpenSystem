/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
// ContextImpl:
//   Implementation-specific functionality associated with a GL Context.
//

#include "libANGLE/renderer/ContextImpl.h"

#include "common/base/anglebase/no_destructor.h"
#include "libANGLE/Context.h"

namespace rx
{
ContextImpl::ContextImpl(const gl::State &state, gl::ErrorSet *errorSet)
    : mState(state), mMemoryProgramCache(nullptr), mErrors(errorSet)
{}

ContextImpl::~ContextImpl() {}

void ContextImpl::invalidateTexture(gl::TextureType target)
{
    UNREACHABLE();
}

angle::Result ContextImpl::startTiling(const gl::Context *context,
                                       const gl::Rectangle &area,
                                       GLbitfield preserveMask)
{
    UNREACHABLE();
    return angle::Result::Stop;
}

angle::Result ContextImpl::endTiling(const gl::Context *context, GLbitfield preserveMask)
{
    UNREACHABLE();
    return angle::Result::Stop;
}

angle::Result ContextImpl::onUnMakeCurrent(const gl::Context *context)
{
    return angle::Result::Continue;
}

angle::Result ContextImpl::handleNoopDrawEvent()
{
    return angle::Result::Continue;
}

void ContextImpl::setMemoryProgramCache(gl::MemoryProgramCache *memoryProgramCache)
{
    mMemoryProgramCache = memoryProgramCache;
}

void ContextImpl::handleError(GLenum errorCode,
                              const char *message,
                              const char *file,
                              const char *function,
                              unsigned int line)
{
    std::stringstream errorStream;
    errorStream << "Internal error: " << gl::FmtHex(errorCode) << ": " << message;
    mErrors->handleError(errorCode, errorStream.str().c_str(), file, function, line);
}

egl::ContextPriority ContextImpl::getContextPriority() const
{
    return egl::ContextPriority::Medium;
}

egl::Error ContextImpl::releaseHighPowerGPU(gl::Context *)
{
    return egl::NoError();
}

egl::Error ContextImpl::reacquireHighPowerGPU(gl::Context *)
{
    return egl::NoError();
}

void ContextImpl::acquireExternalContext(const gl::Context *context) {}

void ContextImpl::releaseExternalContext(const gl::Context *context) {}

angle::Result ContextImpl::acquireTextures(const gl::Context *context,
                                           const gl::TextureBarrierVector &textureBarriers)
{
    UNREACHABLE();
    return angle::Result::Stop;
}

angle::Result ContextImpl::releaseTextures(const gl::Context *context,
                                           gl::TextureBarrierVector *textureBarriers)
{
    UNREACHABLE();
    return angle::Result::Stop;
}

const angle::PerfMonitorCounterGroups &ContextImpl::getPerfMonitorCounters()
{
    static angle::base::NoDestructor<angle::PerfMonitorCounterGroups> sCounters;
    return *sCounters;
}

angle::Result ContextImpl::bindMetalRasterizationRateMap(gl::Context *,
                                                         RenderbufferImpl *renderbuffer,
                                                         GLMTLRasterizationRateMapANGLE map)
{
    UNREACHABLE();
    return angle::Result::Stop;
}

}  // namespace rx
