/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// LinuxWindow.cpp: Implementation of OSWindow::New for Linux

#include "util/OSWindow.h"

#if defined(ANGLE_USE_WAYLAND)
#    include "wayland/WaylandWindow.h"
#endif

#if defined(ANGLE_USE_X11)
#    include "x11/X11Window.h"
#endif

// static
#if defined(ANGLE_USE_X11) || defined(ANGLE_USE_WAYLAND)
OSWindow *OSWindow::New()
{
#    if defined(ANGLE_USE_X11)
    // Prefer X11
    if (IsX11WindowAvailable())
    {
        return CreateX11Window();
    }
#    endif

#    if defined(ANGLE_USE_WAYLAND)
    if (IsWaylandWindowAvailable())
    {
        return new WaylandWindow();
    }
#    endif

    return nullptr;
}
#endif
