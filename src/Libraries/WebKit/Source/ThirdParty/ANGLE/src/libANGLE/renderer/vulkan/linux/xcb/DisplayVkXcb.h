/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DisplayVkXcb.h:
//    Defines the class interface for DisplayVkXcb, implementing DisplayVk for X via XCB.
//

#ifndef LIBANGLE_RENDERER_VULKAN_XCB_DISPLAYVKXCB_H_
#define LIBANGLE_RENDERER_VULKAN_XCB_DISPLAYVKXCB_H_

#include "libANGLE/renderer/vulkan/linux/DisplayVkLinux.h"

struct xcb_connection_t;

namespace rx
{

class DisplayVkXcb : public DisplayVkLinux
{
  public:
    DisplayVkXcb(const egl::DisplayState &state);

    egl::Error initialize(egl::Display *display) override;
    void terminate() override;

    bool isValidNativeWindow(EGLNativeWindowType window) const override;

    SurfaceImpl *createWindowSurfaceVk(const egl::SurfaceState &state,
                                       EGLNativeWindowType window) override;

    egl::ConfigSet generateConfigs() override;
    void checkConfigSupport(egl::Config *config) override;

    const char *getWSIExtension() const override;
    angle::Result waitNativeImpl() override;

    angle::NativeWindowSystem getWindowSystem() const override
    {
        return angle::NativeWindowSystem::X11;
    }

  private:
    xcb_connection_t *mXcbConnection;
    // If there is no X Display, obviously it's impossible to connect to it with Xcb,
    // so rendering to windows is not supported, but rendering to pbuffers is still supported.
    // This mode is used in headless ozone testing.
    bool mHasXDisplay;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_XCB_DISPLAYVKXCB_H_
