/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DisplayVkNull.h:
//    Defines the class interface for DisplayVkNull, implementing
//    DisplayVk that doesn't depend on Vulkan extensions. It cannot be
//    used to output any content to a surface.
//

#ifndef LIBANGLE_RENDERER_VULKAN_NULL_DISPLAYVKNULL_H_
#define LIBANGLE_RENDERER_VULKAN_NULL_DISPLAYVKNULL_H_

#include "libANGLE/renderer/vulkan/DisplayVk.h"

namespace rx
{

class DisplayVkNull : public DisplayVk
{
  public:
    DisplayVkNull(const egl::DisplayState &state);

    bool isValidNativeWindow(EGLNativeWindowType window) const override;

    SurfaceImpl *createWindowSurfaceVk(const egl::SurfaceState &state,
                                       EGLNativeWindowType window) override;

    virtual const char *getWSIExtension() const override;
    egl::ConfigSet generateConfigs() override;
    void checkConfigSupport(egl::Config *config) override;

  private:
    std::vector<VkSurfaceFormatKHR> mSurfaceFormats;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_NULL_DISPLAYVKNULL_H_
