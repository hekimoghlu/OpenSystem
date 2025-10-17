/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "angle_test_platform.h"

#include "common/platform.h"
#include "gpu_info_util/SystemInfo.h"

#if defined(ANGLE_PLATFORM_WINDOWS)
#    include <VersionHelpers.h>
#endif  // defined(ANGLE_PLATFORM_WINDOWS)

using namespace angle;

bool IsAdreno()
{
    std::string rendererString(reinterpret_cast<const char *>(glGetString(GL_RENDERER)));
    return (rendererString.find("Adreno") != std::string::npos);
}

bool IsD3D11()
{
    std::string rendererString(reinterpret_cast<const char *>(glGetString(GL_RENDERER)));
    return (rendererString.find("Direct3D11 vs_5_0") != std::string::npos);
}

bool IsD3D9()
{
    std::string rendererString(reinterpret_cast<const char *>(glGetString(GL_RENDERER)));
    return (rendererString.find("Direct3D9") != std::string::npos);
}

bool IsDesktopOpenGL()
{
    return IsOpenGL() && !IsOpenGLES();
}

bool IsOpenGLES()
{
    std::string rendererString(reinterpret_cast<const char *>(glGetString(GL_RENDERER)));
    return (rendererString.find("OpenGL ES") != std::string::npos);
}

bool IsOpenGL()
{
    std::string rendererString(reinterpret_cast<const char *>(glGetString(GL_RENDERER)));
    return (rendererString.find("OpenGL") != std::string::npos);
}

bool IsNULL()
{
    std::string rendererString(reinterpret_cast<const char *>(glGetString(GL_RENDERER)));
    return (rendererString.find("NULL") != std::string::npos);
}

bool IsVulkan()
{
    const char *renderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
    std::string rendererString(renderer);
    return (rendererString.find("Vulkan") != std::string::npos);
}

bool IsMetal()
{
    const char *renderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
    std::string rendererString(renderer);
    return (rendererString.find("ANGLE Metal") != std::string::npos);
}

bool IsD3D()
{
    return IsD3D9() || IsD3D11();
}

bool IsDebug()
{
#if !defined(NDEBUG)
    return true;
#else
    return false;
#endif
}

bool IsRelease()
{
    return !IsDebug();
}

bool EnsureGLExtensionEnabled(const std::string &extName)
{
    if (IsGLExtensionEnabled("GL_ANGLE_request_extension") && IsGLExtensionRequestable(extName))
    {
        glRequestExtensionANGLE(extName.c_str());
    }

    return IsGLExtensionEnabled(extName);
}

bool IsEGLClientExtensionEnabled(const std::string &extName)
{
    return CheckExtensionExists(eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS), extName);
}

bool IsEGLDeviceExtensionEnabled(EGLDeviceEXT device, const std::string &extName)
{
    return CheckExtensionExists(eglQueryDeviceStringEXT(device, EGL_EXTENSIONS), extName);
}

bool IsEGLDisplayExtensionEnabled(EGLDisplay display, const std::string &extName)
{
    return CheckExtensionExists(eglQueryString(display, EGL_EXTENSIONS), extName);
}

bool IsGLExtensionEnabled(const std::string &extName)
{
    return CheckExtensionExists(reinterpret_cast<const char *>(glGetString(GL_EXTENSIONS)),
                                extName);
}

bool IsGLExtensionRequestable(const std::string &extName)
{
    return CheckExtensionExists(
        reinterpret_cast<const char *>(glGetString(GL_REQUESTABLE_EXTENSIONS_ANGLE)), extName);
}
