/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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

// X11Pixmap.cpp: Implementation of OSPixmap for X11

#include "util/linux/x11/X11Pixmap.h"

X11Pixmap::X11Pixmap() : mPixmap(0), mDisplay(nullptr) {}

X11Pixmap::~X11Pixmap()
{
    if (mPixmap)
    {
        XFreePixmap(mDisplay, mPixmap);
    }
}

bool X11Pixmap::initialize(EGLNativeDisplayType display,
                           size_t width,
                           size_t height,
                           int nativeVisual)
{
    mDisplay = reinterpret_cast<Display *>(display);

    int screen  = DefaultScreen(mDisplay);
    Window root = RootWindow(mDisplay, screen);
    int depth   = 0;

    XVisualInfo visualTemplate;
    visualTemplate.visualid = nativeVisual;

    int numVisuals    = 0;
    XVisualInfo *info = XGetVisualInfo(mDisplay, VisualIDMask, &visualTemplate, &numVisuals);
    if (numVisuals == 1)
    {
        depth = info->depth;
    }
    XFree(info);

    mPixmap = XCreatePixmap(mDisplay, root, static_cast<unsigned int>(width),
                            static_cast<unsigned int>(height), depth);

    return mPixmap != 0;
}

EGLNativePixmapType X11Pixmap::getNativePixmap() const
{
    return mPixmap;
}

OSPixmap *CreateOSPixmap()
{
    return new X11Pixmap();
}
