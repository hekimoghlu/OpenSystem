/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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

// FunctionsEGLDL.cpp: Implements the FunctionsEGLDL class.

#include "libANGLE/renderer/gl/egl/FunctionsEGLDL.h"

#include <dlfcn.h>

namespace rx
{
namespace
{
// In ideal world, we would want this to be a member of FunctionsEGLDL,
// and call dlclose() on it in ~FunctionsEGLDL().
// However, some GL implementations are broken and don't allow multiple
// load/unload cycles, but only static linking with them.
// That's why we dlopen() this handle once and never dlclose() it.
// This is consistent with Chromium's CleanupNativeLibraries() code,
// referencing crbug.com/250813 and http://www.xfree86.org/4.3.0/DRI11.html
void *nativeEGLHandle;
}  // anonymous namespace

FunctionsEGLDL::FunctionsEGLDL() : mGetProcAddressPtr(nullptr) {}

FunctionsEGLDL::~FunctionsEGLDL() {}

egl::Error FunctionsEGLDL::initialize(EGLAttrib platformType,
                                      EGLNativeDisplayType nativeDisplay,
                                      const char *libName,
                                      void *eglHandle)
{

    if (eglHandle)
    {
        // If the handle is provided, use it.
        // Caller has already dlopened the vendor library.
        nativeEGLHandle = eglHandle;
    }

    if (!nativeEGLHandle)
    {
        nativeEGLHandle = dlopen(libName, RTLD_NOW);
        if (!nativeEGLHandle)
        {
            return egl::EglNotInitialized() << "Could not dlopen native EGL: " << dlerror();
        }
    }

    mGetProcAddressPtr =
        reinterpret_cast<PFNEGLGETPROCADDRESSPROC>(dlsym(nativeEGLHandle, "eglGetProcAddress"));
    if (!mGetProcAddressPtr)
    {
        return egl::EglNotInitialized() << "Could not find eglGetProcAddress";
    }

    return FunctionsEGL::initialize(platformType, nativeDisplay);
}

void *FunctionsEGLDL::getProcAddress(const char *name) const
{
    void *f = reinterpret_cast<void *>(mGetProcAddressPtr(name));
    if (f)
    {
        return f;
    }
    return dlsym(nativeEGLHandle, name);
}

}  // namespace rx
