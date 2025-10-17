/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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

#ifndef LIBANGLE_RENDERER_D3D_D3D9_DEBUGANNOTATOR9_H_
#define LIBANGLE_RENDERER_D3D_D3D9_DEBUGANNOTATOR9_H_

#include "libANGLE/LoggingAnnotator.h"

namespace rx
{

class DebugAnnotator9 : public angle::LoggingAnnotator
{
  public:
    DebugAnnotator9() {}
    void beginEvent(gl::Context *context,
                    angle::EntryPoint entryPoint,
                    const char *eventName,
                    const char *eventMessage) override;
    void endEvent(gl::Context *context,
                  const char *eventName,
                  angle::EntryPoint entryPoint) override;
    void setMarker(gl::Context *context, const char *markerName) override;
    bool getStatus(const gl::Context *context) override;

  private:
    static constexpr size_t kMaxMessageLength = 256;
    wchar_t mWCharMessage[kMaxMessageLength];
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D9_DEBUGANNOTATOR9_H_
