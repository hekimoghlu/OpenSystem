/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// DisplayWindow.cpp: Implementation of OSWindow for Linux Display

#include "util/display/DisplayWindow.h"

#include "common/debug.h"
#include "util/Timer.h"
#include "util/test_utils.h"

DisplayWindow::DisplayWindow()
{
    mWindow.width  = 0;
    mWindow.height = 0;
}

DisplayWindow::~DisplayWindow() {}

bool DisplayWindow::initializeImpl(const std::string &name, int width, int height)
{
    return resize(width, height);
}

void DisplayWindow::disableErrorMessageDialog() {}

void DisplayWindow::destroy() {}

void DisplayWindow::resetNativeWindow() {}

EGLNativeWindowType DisplayWindow::getNativeWindow() const
{
    return (EGLNativeWindowType)&mWindow;
}

EGLNativeDisplayType DisplayWindow::getNativeDisplay() const
{
    return NULL;
}

void DisplayWindow::messageLoop() {}

void DisplayWindow::setMousePosition(int x, int y)
{
    UNIMPLEMENTED();
}

bool DisplayWindow::setOrientation(int width, int height)
{
    UNIMPLEMENTED();
    return true;
}

bool DisplayWindow::setPosition(int x, int y)
{
    UNIMPLEMENTED();
    return true;
}

bool DisplayWindow::resize(int width, int height)
{
    mWindow.width  = width;
    mWindow.height = height;
    return true;
}

void DisplayWindow::setVisible(bool isVisible) {}

void DisplayWindow::signalTestEvent()
{
    Event event;
    event.Type   = Event::EVENT_TEST;
    event.Move.X = 0;
    event.Move.Y = 0;
    pushEvent(event);
}

// static
#if defined(ANGLE_USE_VULKAN_DISPLAY) && defined(EGL_NO_X11) && !defined(ANGLE_USE_WAYLAND)
OSWindow *OSWindow::New()
{
    return new DisplayWindow();
}
#endif
