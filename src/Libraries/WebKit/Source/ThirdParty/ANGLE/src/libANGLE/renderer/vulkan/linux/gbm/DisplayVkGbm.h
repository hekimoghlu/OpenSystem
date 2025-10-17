/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
// DisplayVkGbm.h:
//    Defines the class interface for DisplayVkGbm, implementing DisplayVk for GBM.
//

#ifndef LIBANGLE_RENDERER_VULKAN_GBM_DISPLAYVKGBM_H_
#define LIBANGLE_RENDERER_VULKAN_GBM_DISPLAYVKGBM_H_

#include "libANGLE/renderer/vulkan/linux/DisplayVkLinux.h"

struct gbm_device;

namespace rx
{

class DisplayVkGbm : public DisplayVkLinux
{
  public:
    DisplayVkGbm(const egl::DisplayState &state);

    egl::Error initialize(egl::Display *display) override;
    void terminate() override;

    bool isValidNativeWindow(EGLNativeWindowType window) const override;

    SurfaceImpl *createWindowSurfaceVk(const egl::SurfaceState &state,
                                       EGLNativeWindowType window) override;

    egl::ConfigSet generateConfigs() override;
    void checkConfigSupport(egl::Config *config) override;

    const char *getWSIExtension() const override;
    angle::NativeWindowSystem getWindowSystem() const override;

  private:
    gbm_device *mGbmDevice;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_GBM_DISPLAYVKGBM_H_
