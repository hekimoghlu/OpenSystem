/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// WGLWindow:
//   Implements initializing a WGL rendering context.
//

#ifndef UTIL_WINDOWS_WGLWINDOW_H_
#define UTIL_WINDOWS_WGLWINDOW_H_

#include "common/angleutils.h"
#include "export.h"
#include "util/EGLWindow.h"

class OSWindow;

namespace angle
{
class Library;
}  // namespace angle

class ANGLE_UTIL_EXPORT WGLWindow : public GLWindowBase
{
  public:
    static WGLWindow *New(int majorVersion, int minorVersion);
    static void Delete(WGLWindow **window);

    // Internally initializes GL resources.
    bool initializeGL(OSWindow *osWindow,
                      angle::Library *glWindowingLibrary,
                      angle::GLESDriverType driverType,
                      const EGLPlatformParameters &platformParams,
                      const ConfigParameters &configParams) override;

    GLWindowResult initializeGLWithResult(OSWindow *osWindow,
                                          angle::Library *glWindowingLibrary,
                                          angle::GLESDriverType driverType,
                                          const EGLPlatformParameters &platformParams,
                                          const ConfigParameters &configParams) override;

    void destroyGL() override;
    bool isGLInitialized() const override;
    bool makeCurrent() override;
    void swap() override;
    bool hasError() const override;
    bool setSwapInterval(EGLint swapInterval) override;
    angle::GenericProc getProcAddress(const char *name) override;
    // Initializes WGL resources.
    GLWindowContext getCurrentContextGeneric() override;
    GLWindowContext createContextGeneric(GLWindowContext share) override;
    bool makeCurrentGeneric(GLWindowContext context) override;
    Image createImage(GLWindowContext context,
                      Enum target,
                      ClientBuffer buffer,
                      const Attrib *attrib_list) override;
    Image createImageKHR(GLWindowContext context,
                         Enum target,
                         ClientBuffer buffer,
                         const AttribKHR *attrib_list) override;
    EGLBoolean destroyImage(Image image) override;
    EGLBoolean destroyImageKHR(Image image) override;
    Sync createSync(EGLDisplay dpy, EGLenum type, const EGLAttrib *attrib_list) override;
    Sync createSyncKHR(EGLDisplay dpy, EGLenum type, const EGLint *attrib_list) override;
    EGLBoolean destroySync(EGLDisplay dpy, Sync sync) override;
    EGLBoolean destroySyncKHR(EGLDisplay dpy, Sync sync) override;
    EGLint clientWaitSync(EGLDisplay dpy, Sync sync, EGLint flags, EGLTimeKHR timeout) override;
    EGLint clientWaitSyncKHR(EGLDisplay dpy, Sync sync, EGLint flags, EGLTimeKHR timeout) override;
    EGLint getEGLError() override;
    Display getCurrentDisplay() override;
    Surface createPbufferSurface(const EGLint *attrib_list) override;
    EGLBoolean destroySurface(Surface surface) override;

    EGLBoolean bindTexImage(EGLSurface surface, EGLint buffer) override;
    EGLBoolean releaseTexImage(EGLSurface surface, EGLint buffer) override;
    bool makeCurrent(EGLSurface draw, EGLSurface read, EGLContext context) override;

    // Create a WGL context with this window's configuration
    HGLRC createContext(const ConfigParameters &configParams, HGLRC shareContext);
    // Make the WGL context current
    bool makeCurrent(HGLRC context);

  private:
    WGLWindow(int majorVersion, int minorVersion);
    ~WGLWindow() override;

    // OS resources.
    HDC mDeviceContext;
    HGLRC mWGLContext;
    HWND mWindow;
};

#endif  // UTIL_WINDOWS_WGLWINDOW_H_
