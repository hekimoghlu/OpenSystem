/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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

// NativeWindowD3D.h: Defines NativeWindowD3D, a class for managing and performing operations on an
// EGLNativeWindowType for the D3D renderers.

#ifndef LIBANGLE_RENDERER_D3D_NATIVEWINDOWD3D_H_
#define LIBANGLE_RENDERER_D3D_NATIVEWINDOWD3D_H_

#include "common/debug.h"
#include "common/platform.h"

#include <EGL/eglplatform.h>
#include "libANGLE/Config.h"

namespace rx
{
class NativeWindowD3D : angle::NonCopyable
{
  public:
    NativeWindowD3D(EGLNativeWindowType window);
    virtual ~NativeWindowD3D();

    virtual bool initialize()                     = 0;
    virtual bool getClientRect(LPRECT rect) const = 0;
    virtual bool isIconic() const                 = 0;

    inline EGLNativeWindowType getNativeWindow() const { return mWindow; }

  private:
    EGLNativeWindowType mWindow;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_NATIVEWINDOWD3D_H_
