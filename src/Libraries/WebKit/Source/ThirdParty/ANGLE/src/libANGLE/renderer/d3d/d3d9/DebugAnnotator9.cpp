/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
// DebugAnnotator9.h: D3D9 helpers for adding trace annotations.
//

#include "libANGLE/renderer/d3d/d3d9/DebugAnnotator9.h"

#include "common/platform.h"

namespace rx
{

void DebugAnnotator9::beginEvent(gl::Context *context,
                                 angle::EntryPoint entryPoint,
                                 const char *eventName,
                                 const char *eventMessage)
{
    angle::LoggingAnnotator::beginEvent(context, entryPoint, eventName, eventMessage);
    std::mbstate_t state = std::mbstate_t();
    std::mbsrtowcs(mWCharMessage, &eventMessage, kMaxMessageLength, &state);
    D3DPERF_BeginEvent(0, mWCharMessage);
}

void DebugAnnotator9::endEvent(gl::Context *context,
                               const char *eventName,
                               angle::EntryPoint entryPoint)
{
    angle::LoggingAnnotator::endEvent(context, eventName, entryPoint);
    D3DPERF_EndEvent();
}

void DebugAnnotator9::setMarker(gl::Context *context, const char *markerName)
{
    angle::LoggingAnnotator::setMarker(context, markerName);
    std::mbstate_t state = std::mbstate_t();
    std::mbsrtowcs(mWCharMessage, &markerName, kMaxMessageLength, &state);
    D3DPERF_SetMarker(0, mWCharMessage);
}

bool DebugAnnotator9::getStatus(const gl::Context *context)
{
    return !!D3DPERF_GetStatus();
}

}  // namespace rx
