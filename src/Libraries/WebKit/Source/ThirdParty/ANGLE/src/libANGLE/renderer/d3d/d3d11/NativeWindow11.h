/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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

// NativeWindow11.h: Defines NativeWindow11, a class for managing and performing operations on an
// EGLNativeWindowType for the D3D11 renderer.

#ifndef LIBANGLE_RENDERER_D3D_D3D11_NATIVEWINDOW11_H_
#define LIBANGLE_RENDERER_D3D_D3D11_NATIVEWINDOW11_H_

#include "common/debug.h"
#include "common/platform.h"

#include "libANGLE/Config.h"
#include "libANGLE/renderer/d3d/NativeWindowD3D.h"

namespace rx
{

class NativeWindow11 : public NativeWindowD3D
{
  public:
    NativeWindow11(EGLNativeWindowType window) : NativeWindowD3D(window) {}

    virtual HRESULT createSwapChain(ID3D11Device *device,
                                    IDXGIFactory *factory,
                                    DXGI_FORMAT format,
                                    UINT width,
                                    UINT height,
                                    UINT samples,
                                    IDXGISwapChain **swapChain) = 0;
    virtual void commitChange()                                 = 0;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D11_NATIVEWINDOW11_H_
