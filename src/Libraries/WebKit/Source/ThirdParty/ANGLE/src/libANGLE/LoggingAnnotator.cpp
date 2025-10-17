/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// LoggingAnnotator.cpp: DebugAnnotator implementing logging
//

#include "libANGLE/LoggingAnnotator.h"

#include "libANGLE/trace.h"

namespace angle
{

bool LoggingAnnotator::getStatus(const gl::Context *context)
{
    return false;
}

void LoggingAnnotator::beginEvent(gl::Context *context,
                                  EntryPoint entryPoint,
                                  const char *eventName,
                                  const char *eventMessage)
{
    ANGLE_TRACE_EVENT_BEGIN0("gpu.angle", eventName);
}

void LoggingAnnotator::endEvent(gl::Context *context, const char *eventName, EntryPoint entryPoint)
{
    ANGLE_TRACE_EVENT_END0("gpu.angle", eventName);
}

void LoggingAnnotator::setMarker(gl::Context *context, const char *markerName)
{
    ANGLE_TRACE_EVENT_INSTANT0("gpu.angle", markerName);
}

void LoggingAnnotator::logMessage(const gl::LogMessage &msg) const
{
    auto *plat = ANGLEPlatformCurrent();
    if (plat != nullptr)
    {
        switch (msg.getSeverity())
        {
            case gl::LOG_FATAL:
            case gl::LOG_ERR:
                plat->logError(plat, msg.getMessage().c_str());
                break;
            case gl::LOG_WARN:
                plat->logWarning(plat, msg.getMessage().c_str());
                break;
            case gl::LOG_INFO:
                plat->logInfo(plat, msg.getMessage().c_str());
                break;
            default:
                UNREACHABLE();
        }
    }
    gl::Trace(msg.getSeverity(), msg.getMessage().c_str());
}

}  // namespace angle
