/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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

// IOSWindow.mm: Implementation of OSWindow for iOS

#include "util/ios/IOSWindow.h"

#include <set>

#include "anglebase/no_destructor.h"
#include "common/debug.h"

#import <UIKit/UIKit.h>

static CALayer *rootLayer()
{
    return [[[[[UIApplication sharedApplication] delegate] window] rootViewController] view].layer;
}

bool IOSWindow::initializeImpl(const std::string &name, int width, int height)
{
    resize(width, height);
    return true;
}

EGLNativeWindowType IOSWindow::getNativeWindow() const
{
    return rootLayer();
}

bool IOSWindow::setOrientation(int width, int height)
{
    UNIMPLEMENTED();
    return false;
}

bool IOSWindow::resize(int width, int height)
{
    rootLayer().frame = CGRectMake(0, 0, width, height);
    return true;
}

// static
OSWindow *OSWindow::New()
{
    return new IOSWindow;
}
