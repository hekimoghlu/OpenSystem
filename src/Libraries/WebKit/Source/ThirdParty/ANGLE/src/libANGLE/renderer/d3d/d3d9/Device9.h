/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Device9.h: D3D9 implementation of egl::Device

#ifndef LIBANGLE_RENDERER_D3D_D3D9_DEVICE9_H_
#define LIBANGLE_RENDERER_D3D_D3D9_DEVICE9_H_

#include "libANGLE/Device.h"
#include "libANGLE/renderer/DeviceImpl.h"
#include "libANGLE/renderer/d3d/d3d9/Renderer9.h"

namespace rx
{
class Device9 : public DeviceImpl
{
  public:
    Device9(IDirect3DDevice9 *device);
    ~Device9() override;

    egl::Error initialize() override;
    egl::Error getAttribute(const egl::Display *display,
                            EGLint attribute,
                            void **outValue) override;
    void generateExtensions(egl::DeviceExtensions *outExtensions) const override;

  private:
    IDirect3DDevice9 *mDevice;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D9_DEVICE9_H_
