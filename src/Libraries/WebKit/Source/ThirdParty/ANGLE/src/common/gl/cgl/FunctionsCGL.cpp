/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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

// FunctionsCGL.cpp: Exposing the soft-linked CGL interface.

#include "common/gl/cgl/FunctionsCGL.h"
#include "common/platform.h"

SOFT_LINK_FRAMEWORK_SOURCE(OpenGL)

SOFT_LINK_FUNCTION_SOURCE(OpenGL,
                          CGLChoosePixelFormat,
                          CGLError,
                          (const CGLPixelFormatAttribute *attribs,
                           CGLPixelFormatObj *pix,
                           GLint *npix),
                          (attribs, pix, npix))
SOFT_LINK_FUNCTION_SOURCE(OpenGL,
                          CGLCreateContext,
                          CGLError,
                          (CGLPixelFormatObj pix, CGLContextObj share, CGLContextObj *ctx),
                          (pix, share, ctx))
SOFT_LINK_FUNCTION_SOURCE(
    OpenGL,
    CGLDescribePixelFormat,
    CGLError,
    (CGLPixelFormatObj pix, GLint pix_num, CGLPixelFormatAttribute attrib, GLint *value),
    (pix, pix_num, attrib, value))
SOFT_LINK_FUNCTION_SOURCE(OpenGL, CGLDestroyContext, CGLError, (CGLContextObj ctx), (ctx))
SOFT_LINK_FUNCTION_SOURCE(OpenGL, CGLDestroyPixelFormat, CGLError, (CGLPixelFormatObj pix), (pix))
SOFT_LINK_FUNCTION_SOURCE(OpenGL, CGLErrorString, const char *, (CGLError error), (error))
SOFT_LINK_FUNCTION_SOURCE(OpenGL, CGLReleaseContext, void, (CGLContextObj ctx), (ctx))
SOFT_LINK_FUNCTION_SOURCE(OpenGL, CGLGetCurrentContext, CGLContextObj, (void), ())
SOFT_LINK_FUNCTION_SOURCE(OpenGL, CGLSetCurrentContext, CGLError, (CGLContextObj ctx), (ctx))
SOFT_LINK_FUNCTION_SOURCE(OpenGL,
                          CGLSetVirtualScreen,
                          CGLError,
                          (CGLContextObj ctx, GLint screen),
                          (ctx, screen))
SOFT_LINK_FUNCTION_SOURCE(
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
SOFT_LINK_FUNCTION_SOURCE(OpenGL, CGLUpdateContext, CGLError, (CGLContextObj ctx), (ctx))

SOFT_LINK_FUNCTION_SOURCE(
    OpenGL,
    CGLDescribeRenderer,
    CGLError,
    (CGLRendererInfoObj rend, GLint rend_num, CGLRendererProperty prop, GLint *value),
    (rend, rend_num, prop, value))
SOFT_LINK_FUNCTION_SOURCE(OpenGL,
                          CGLDestroyRendererInfo,
                          CGLError,
                          (CGLRendererInfoObj rend),
                          (rend))
SOFT_LINK_FUNCTION_SOURCE(OpenGL,
                          CGLQueryRendererInfo,
                          CGLError,
                          (GLuint display_mask, CGLRendererInfoObj *rend, GLint *nrend),
                          (display_mask, rend, nrend))
