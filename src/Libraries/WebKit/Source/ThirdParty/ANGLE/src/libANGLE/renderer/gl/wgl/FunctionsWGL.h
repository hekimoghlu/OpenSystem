/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FunctionsWGL.h: Defines the FuntionsWGL class to contain loaded WGL functions

#ifndef LIBANGLE_RENDERER_GL_WGL_FUNCTIONS_WGL
#define LIBANGLE_RENDERER_GL_WGL_FUNCTIONS_WGL

#include "common/angleutils.h"
#include "libANGLE/renderer/gl/wgl/functionswgl_typedefs.h"

namespace rx
{

class FunctionsWGL : angle::NonCopyable
{
  public:
    FunctionsWGL();
    ~FunctionsWGL();

    // Loads all available wgl functions, may be called multiple times
    void initialize(HMODULE glModule, HDC context);

    // Extension information
    std::vector<std::string> extensions;
    bool hasExtension(const std::string &ext) const;

    // Base WGL functions
    PFNWGLCOPYCONTEXTPROC copyContext;
    PFNWGLCREATECONTEXTPROC createContext;
    PFNWGLCREATELAYERCONTEXTPROC createLayerContext;
    PFNWGLDELETECONTEXTPROC deleteContext;
    PFNWGLGETCURRENTCONTEXTPROC getCurrentContext;
    PFNWGLGETCURRENTDCPROC getCurrentDC;
    PFNWGLGETPROCADDRESSPROC getProcAddress;
    PFNWGLMAKECURRENTPROC makeCurrent;
    PFNWGLSHARELISTSPROC shareLists;
    PFNWGLUSEFONTBITMAPSAPROC useFontBitmapsA;
    PFNWGLUSEFONTBITMAPSWPROC useFontBitmapsW;
    PFNSWAPBUFFERSPROC swapBuffers;
    PFNWGLUSEFONTOUTLINESAPROC useFontOutlinesA;
    PFNWGLUSEFONTOUTLINESWPROC useFontOutlinesW;
    PFNWGLDESCRIBELAYERPLANEPROC describeLayerPlane;
    PFNWGLSETLAYERPALETTEENTRIESPROC setLayerPaletteEntries;
    PFNWGLGETLAYERPALETTEENTRIESPROC getLayerPaletteEntries;
    PFNWGLREALIZELAYERPALETTEPROC realizeLayerPalette;
    PFNWGLSWAPLAYERBUFFERSPROC swapLayerBuffers;
    PFNWGLSWAPMULTIPLEBUFFERSPROC swapMultipleBuffers;

    // WGL_EXT_extensions_string
    PFNWGLGETEXTENSIONSSTRINGEXTPROC getExtensionStringEXT;

    // WGL_ARB_extensions_string
    PFNWGLGETEXTENSIONSSTRINGARBPROC getExtensionStringARB;

    // WGL_ARB_create_context
    PFNWGLCREATECONTEXTATTRIBSARBPROC createContextAttribsARB;

    // WGL_ARB_pixel_format
    PFNWGLGETPIXELFORMATATTRIBIVARBPROC getPixelFormatAttribivARB;
    PFNWGLGETPIXELFORMATATTRIBFVARBPROC getPixelFormatAttribfvARB;
    PFNWGLCHOOSEPIXELFORMATARBPROC choosePixelFormatARB;

    // WGL_EXT_swap_control
    PFNWGLSWAPINTERVALEXTPROC swapIntervalEXT;

    // WGL_ARB_pbuffer
    PFNWGLCREATEPBUFFERARBPROC createPbufferARB;
    PFNWGLGETPBUFFERDCARBPROC getPbufferDCARB;
    PFNWGLRELEASEPBUFFERDCARBPROC releasePbufferDCARB;
    PFNWGLDESTROYPBUFFERARBPROC destroyPbufferARB;
    PFNWGLQUERYPBUFFERARBPROC queryPbufferARB;

    // WGL_ARB_render_texture
    PFNWGLBINDTEXIMAGEARBPROC bindTexImageARB;
    PFNWGLRELEASETEXIMAGEARBPROC releaseTexImageARB;
    PFNWGLSETPBUFFERATTRIBARBPROC setPbufferAttribARB;

    // WGL_NV_DX_interop
    PFNWGLDXSETRESOURCESHAREHANDLENVPROC dxSetResourceShareHandleNV;
    PFNWGLDXOPENDEVICENVPROC dxOpenDeviceNV;
    PFNWGLDXCLOSEDEVICENVPROC dxCloseDeviceNV;
    PFNWGLDXREGISTEROBJECTNVPROC dxRegisterObjectNV;
    PFNWGLDXUNREGISTEROBJECTNVPROC dxUnregisterObjectNV;
    PFNWGLDXOBJECTACCESSNVPROC dxObjectAccessNV;
    PFNWGLDXLOCKOBJECTSNVPROC dxLockObjectsNV;
    PFNWGLDXUNLOCKOBJECTSNVPROC dxUnlockObjectsNV;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_WGL_FUNCTIONS_WGL
