/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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
//

// FunctionsCGL.h: Exposing the soft-linked CGL interface.

#ifndef CGL_FUNCTIONS_H_
#define CGL_FUNCTIONS_H_

#include <OpenGL/OpenGL.h>

#include "common/apple/SoftLinking.h"

SOFT_LINK_FRAMEWORK_HEADER(OpenGL)

SOFT_LINK_FUNCTION_HEADER(OpenGL,
                          CGLChoosePixelFormat,
                          CGLError,
                          (const CGLPixelFormatAttribute *attribs,
                           CGLPixelFormatObj *pix,
                           GLint *npix),
                          (attribs, pix, npix))
SOFT_LINK_FUNCTION_HEADER(OpenGL,
                          CGLCreateContext,
                          CGLError,
                          (CGLPixelFormatObj pix, CGLContextObj share, CGLContextObj *ctx),
                          (pix, share, ctx))
SOFT_LINK_FUNCTION_HEADER(
    OpenGL,
    CGLDescribePixelFormat,
    CGLError,
    (CGLPixelFormatObj pix, GLint pix_num, CGLPixelFormatAttribute attrib, GLint *value),
    (pix, pix_num, attrib, value))
SOFT_LINK_FUNCTION_HEADER(
    OpenGL,
    CGLDescribeRenderer,
    CGLError,
    (CGLRendererInfoObj rend, GLint rend_num, CGLRendererProperty prop, GLint *value),
    (rend, rend_num, prop, value))
SOFT_LINK_FUNCTION_HEADER(OpenGL, CGLDestroyContext, CGLError, (CGLContextObj ctx), (ctx))
SOFT_LINK_FUNCTION_HEADER(OpenGL, CGLDestroyPixelFormat, CGLError, (CGLPixelFormatObj pix), (pix))
SOFT_LINK_FUNCTION_HEADER(OpenGL,
                          CGLDestroyRendererInfo,
                          CGLError,
                          (CGLRendererInfoObj rend),
                          (rend))
SOFT_LINK_FUNCTION_HEADER(OpenGL, CGLErrorString, const char *, (CGLError error), (error))
SOFT_LINK_FUNCTION_HEADER(OpenGL,
                          CGLQueryRendererInfo,
                          CGLError,
                          (GLuint display_mask, CGLRendererInfoObj *rend, GLint *nrend),
                          (display_mask, rend, nrend))
SOFT_LINK_FUNCTION_HEADER(OpenGL, CGLReleaseContext, void, (CGLContextObj ctx), (ctx))
SOFT_LINK_FUNCTION_HEADER(OpenGL, CGLGetCurrentContext, CGLContextObj, (void), ())
SOFT_LINK_FUNCTION_HEADER(OpenGL, CGLSetCurrentContext, CGLError, (CGLContextObj ctx), (ctx))
SOFT_LINK_FUNCTION_HEADER(OpenGL,
                          CGLSetVirtualScreen,
                          CGLError,
                          (CGLContextObj ctx, GLint screen),
                          (ctx, screen))
SOFT_LINK_FUNCTION_HEADER(
    OpenGL,
    CGLTexImageIOSurface2D,
    CGLError,
    (CGLContextObj ctx,
     GLenum target,
     GLenum internal_format,
     GLsizei width,
     GLsizei height,
     GLenum format,
     GLenum type,
     IOSurfaceRef ioSurface,
     GLuint plane),
    (ctx, target, internal_format, width, height, format, type, ioSurface, plane))
SOFT_LINK_FUNCTION_HEADER(OpenGL, CGLUpdateContext, CGLError, (CGLContextObj ctx), (ctx))

#endif  // CGL_FUNCTIONS_H_
