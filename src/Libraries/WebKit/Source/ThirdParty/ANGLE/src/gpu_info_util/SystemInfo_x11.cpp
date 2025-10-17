/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 1, 2023.
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
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// SystemInfo_x11.cpp: implementation of the X11-specific parts of SystemInfo.h

#include "gpu_info_util/SystemInfo_internal.h"

#include <X11/Xlib.h>

#include "common/debug.h"
#include "third_party/libXNVCtrl/NVCtrl.h"
#include "third_party/libXNVCtrl/NVCtrlLib.h"

#if !defined(GPU_INFO_USE_X11)
#    error SystemInfo_x11.cpp compiled without GPU_INFO_USE_X11
#endif

namespace angle
{

bool GetNvidiaDriverVersionWithXNVCtrl(std::string *version)
{
    *version = "";

    int eventBase = 0;
    int errorBase = 0;

    Display *display = XOpenDisplay(nullptr);

    if (display && XNVCTRLQueryExtension(display, &eventBase, &errorBase))
    {
        int screenCount = ScreenCount(display);
        for (int screen = 0; screen < screenCount; ++screen)
        {
            char *buffer = nullptr;
            if (XNVCTRLIsNvScreen(display, screen) &&
                XNVCTRLQueryStringAttribute(display, screen, 0,
                                            NV_CTRL_STRING_NVIDIA_DRIVER_VERSION, &buffer))
            {
                ASSERT(buffer != nullptr);
                *version = buffer;
                XFree(buffer);
                return true;
            }
        }
    }

    if (display)
    {
        XCloseDisplay(display);
    }

    return false;
}
}  // namespace angle
