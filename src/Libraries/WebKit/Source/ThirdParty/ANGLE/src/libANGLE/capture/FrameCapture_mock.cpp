/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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
// FrameCapture_mock.cpp:
//   ANGLE mock Frame capture implementation.
//

#include "libANGLE/capture/FrameCapture.h"

#if ANGLE_CAPTURE_ENABLED
#    error Frame capture must be disabled to include this file.
#endif  // ANGLE_CAPTURE_ENABLED

namespace angle
{
CallCapture::~CallCapture() {}
ParamBuffer::~ParamBuffer() {}
ParamCapture::~ParamCapture() {}
ResourceTracker::ResourceTracker() {}
ResourceTracker::~ResourceTracker() {}
TrackedResource::TrackedResource() {}
TrackedResource::~TrackedResource() {}
StateResetHelper::StateResetHelper() {}
StateResetHelper::~StateResetHelper() {}
DataTracker::DataTracker() {}
DataTracker::~DataTracker() {}
DataCounters::DataCounters() {}
DataCounters::~DataCounters() {}
StringCounters::StringCounters() {}
StringCounters::~StringCounters() {}
ReplayWriter::ReplayWriter() {}
ReplayWriter::~ReplayWriter() {}

FrameCapture::FrameCapture() {}
FrameCapture::~FrameCapture() {}

FrameCaptureShared::FrameCaptureShared() : mEnabled(false) {}
FrameCaptureShared::~FrameCaptureShared() {}
void FrameCaptureShared::onEndFrame(gl::Context *context) {}
void FrameCaptureShared::onMakeCurrent(const gl::Context *context, const egl::Surface *drawSurface)
{}
void FrameCaptureShared::onDestroyContext(const gl::Context *context) {}
void *FrameCaptureShared::maybeGetShadowMemoryPointer(gl::Buffer *buffer,
                                                      GLsizeiptr length,
                                                      GLbitfield access)
{
    return buffer->getMapPointer();
}
void FrameCaptureShared::determineMemoryProtectionSupport(gl::Context *context) {}

const ProgramSources &FrameCaptureShared::getProgramSources(gl::ShaderProgramID id) const
{
    const auto &foundSources = mCachedProgramSources.find(id);
    return foundSources->second;
}
void FrameCaptureShared::setProgramSources(gl::ShaderProgramID id, ProgramSources sources) {}

CoherentBufferTracker::CoherentBufferTracker() {}
CoherentBufferTracker::~CoherentBufferTracker() {}
}  // namespace angle
