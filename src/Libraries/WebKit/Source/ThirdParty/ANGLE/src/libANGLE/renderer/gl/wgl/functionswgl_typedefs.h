/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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

// functionswgl_typedefs.h: Typedefs of WGL functions.

#ifndef LIBANGLE_RENDERER_GL_WGL_FUNCTIONSWGLTYPEDEFS_H_
#define LIBANGLE_RENDERER_GL_WGL_FUNCTIONSWGLTYPEDEFS_H_

#include "common/platform.h"

// This header must be included before wglext.h.
#include <angle_gl.h>

#include <GL/wglext.h>

namespace rx
{

typedef BOOL(WINAPI *PFNWGLCOPYCONTEXTPROC)(HGLRC, HGLRC, UINT);
typedef HGLRC(WINAPI *PFNWGLCREATECONTEXTPROC)(HDC);
typedef HGLRC(WINAPI *PFNWGLCREATELAYERCONTEXTPROC)(HDC, int);
typedef BOOL(WINAPI *PFNWGLDELETECONTEXTPROC)(HGLRC);
typedef HGLRC(WINAPI *PFNWGLGETCURRENTCONTEXTPROC)(VOID);
typedef HDC(WINAPI *PFNWGLGETCURRENTDCPROC)(VOID);
typedef PROC(WINAPI *PFNWGLGETPROCADDRESSPROC)(LPCSTR);
typedef BOOL(WINAPI *PFNWGLMAKECURRENTPROC)(HDC, HGLRC);
typedef BOOL(WINAPI *PFNWGLSHARELISTSPROC)(HGLRC, HGLRC);
typedef BOOL(WINAPI *PFNWGLUSEFONTBITMAPSAPROC)(HDC, DWORD, DWORD, DWORD);
typedef BOOL(WINAPI *PFNWGLUSEFONTBITMAPSWPROC)(HDC, DWORD, DWORD, DWORD);
typedef BOOL(WINAPI *PFNSWAPBUFFERSPROC)(HDC);
typedef BOOL(WINAPI *PFNWGLUSEFONTOUTLINESAPROC)(HDC,
                                                 DWORD,
                                                 DWORD,
                                                 DWORD,
                                                 FLOAT,
                                                 FLOAT,
                                                 int,
                                                 LPGLYPHMETRICSFLOAT);
typedef BOOL(WINAPI *PFNWGLUSEFONTOUTLINESWPROC)(HDC,
                                                 DWORD,
                                                 DWORD,
                                                 DWORD,
                                                 FLOAT,
                                                 FLOAT,
                                                 int,
                                                 LPGLYPHMETRICSFLOAT);
typedef BOOL(WINAPI *PFNWGLDESCRIBELAYERPLANEPROC)(HDC, int, int, UINT, LPLAYERPLANEDESCRIPTOR);
typedef int(WINAPI *PFNWGLSETLAYERPALETTEENTRIESPROC)(HDC, int, int, int, CONST COLORREF *);
typedef int(WINAPI *PFNWGLGETLAYERPALETTEENTRIESPROC)(HDC, int, int, int, COLORREF *);
typedef BOOL(WINAPI *PFNWGLREALIZELAYERPALETTEPROC)(HDC, int, BOOL);
typedef BOOL(WINAPI *PFNWGLSWAPLAYERBUFFERSPROC)(HDC, UINT);
typedef DWORD(WINAPI *PFNWGLSWAPMULTIPLEBUFFERSPROC)(UINT, CONST WGLSWAP *);

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_WGL_FUNCTIONSWGLTYPEDEFS_H_
