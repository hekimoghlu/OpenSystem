/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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

// OSXWindow.h: Definition of the implementation of OSWindow for OSX

#ifndef UTIL_OSX_WINDOW_H_
#define UTIL_OSX_WINDOW_H_

#import <Cocoa/Cocoa.h>

#include "util/OSWindow.h"

class OSXWindow;

@interface WindowDelegate : NSObject {
    OSXWindow *mWindow;
}
- (id)initWithWindow:(OSXWindow *)window;
@end

@interface ContentView : NSView {
    OSXWindow *mWindow;
    NSTrackingArea *mTrackingArea;
    int mCurrentModifier;
}
- (id)initWithWindow:(OSXWindow *)window;
@end

class OSXWindow : public OSWindow
{
  public:
    OSXWindow();
    ~OSXWindow() override;

    void disableErrorMessageDialog() override;
    void destroy() override;

    void resetNativeWindow() override;
    EGLNativeWindowType getNativeWindow() const override;
    EGLNativeDisplayType getNativeDisplay() const override;

    void messageLoop() override;

    void setMousePosition(int x, int y) override;
    bool setOrientation(int width, int height) override;
    bool setPosition(int x, int y) override;
    bool resize(int width, int height) override;
    void setVisible(bool isVisible) override;

    void signalTestEvent() override;

    NSWindow *getNSWindow() const;

  private:
    bool initializeImpl(const std::string &name, int width, int height) override;

    NSWindow *mWindow;
    WindowDelegate *mDelegate;
    ContentView *mView;
};

#endif  // UTIL_OSX_WINDOW_H_
