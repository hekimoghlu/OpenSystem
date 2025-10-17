/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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
// DeviceEGL.h:
//    Defines the class interface for DeviceEGL, implementing DeviceImpl.
//

#ifndef LIBANGLE_RENDERER_GL_EGL__DEVICEEGL_H_
#define LIBANGLE_RENDERER_GL_EGL__DEVICEEGL_H_

#include "libANGLE/renderer/DeviceImpl.h"

namespace rx
{

class DisplayEGL;

class DeviceEGL : public DeviceImpl
{
  public:
    DeviceEGL(DisplayEGL *display);
    ~DeviceEGL() override;

    egl::Error initialize() override;
    egl::Error getAttribute(const egl::Display *display,
                            EGLint attribute,
                            void **outValue) override;
    void generateExtensions(egl::DeviceExtensions *outExtensions) const override;
    const std::string getDeviceString(EGLint name) override;

  private:
    bool hasExtension(const char *extension) const;

    DisplayEGL *mDisplay;
    EGLDeviceEXT mDevice;
    std::vector<std::string> mExtensions;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_EGL__DEVICEEGL_H_
