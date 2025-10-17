/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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
// DisplayVkWin32.h:
//    Defines the class interface for DisplayVkWin32, implementing DisplayVk for Windows.
//

#ifndef LIBANGLE_RENDERER_VULKAN_WIN32_DISPLAYVKWIN32_H_
#define LIBANGLE_RENDERER_VULKAN_WIN32_DISPLAYVKWIN32_H_

#include "libANGLE/renderer/vulkan/DisplayVk.h"

namespace rx
{
class DisplayVkWin32 : public DisplayVk
{
  public:
    DisplayVkWin32(const egl::DisplayState &state);
    ~DisplayVkWin32() override;

    egl::Error initialize(egl::Display *display) override;
    void terminate() override;

    bool isValidNativeWindow(EGLNativeWindowType window) const override;

    SurfaceImpl *createWindowSurfaceVk(const egl::SurfaceState &state,
                                       EGLNativeWindowType window) override;

    egl::ConfigSet generateConfigs() override;
    void checkConfigSupport(egl::Config *config) override;

    const char *getWSIExtension() const override;

  private:
    ATOM mWindowClass;
    HWND mMockWindow;
    std::vector<VkSurfaceFormatKHR> mSurfaceFormats;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_WIN32_DISPLAYVKWIN32_H_
