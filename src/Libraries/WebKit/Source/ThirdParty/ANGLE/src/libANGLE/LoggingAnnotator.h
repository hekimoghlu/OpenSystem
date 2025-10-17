/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 30, 2021.
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
// LoggingAnnotator.h: DebugAnnotator implementing logging
//

#ifndef LIBANGLE_LOGGINGANNOTATOR_H_
#define LIBANGLE_LOGGINGANNOTATOR_H_

#include "common/debug.h"

namespace gl
{
class Context;
}  // namespace gl

namespace angle
{

class LoggingAnnotator : public gl::DebugAnnotator
{
  public:
    LoggingAnnotator() {}
    ~LoggingAnnotator() override {}
    void beginEvent(gl::Context *context,
                    EntryPoint entryPoint,
                    const char *eventName,
                    const char *eventMessage) override;
    void endEvent(gl::Context *context, const char *eventName, EntryPoint entryPoint) override;
    void setMarker(gl::Context *context, const char *markerName) override;
    bool getStatus(const gl::Context *context) override;
    void logMessage(const gl::LogMessage &msg) const override;
};

}  // namespace angle

#endif  // LIBANGLE_LOGGINGANNOTATOR_H_
