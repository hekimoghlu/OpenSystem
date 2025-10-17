/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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
#pragma once

#include <wtf/SoftLinking.h>

#if PLATFORM(MAC)

#include <OpenGL/OpenGL.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, OpenGL)

SOFT_LINK_FUNCTION_FOR_HEADER(PAL, OpenGL, CGLChoosePixelFormat, CGLError, (const CGLPixelFormatAttribute *attribs, CGLPixelFormatObj *pix, GLint *npix), (attribs, pix, npix))
#define CGLChoosePixelFormat PAL::softLink_OpenGL_CGLChoosePixelFormat
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, OpenGL, CGLCreateContext, CGLError, (CGLPixelFormatObj pix, CGLContextObj share, CGLContextObj *ctx), (pix, share, ctx))
#define CGLCreateContext PAL::softLink_OpenGL_CGLCreateContext
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, OpenGL, CGLDescribePixelFormat, CGLError, (CGLPixelFormatObj pix, GLint pix_num, CGLPixelFormatAttribute attrib, GLint *value), (pix, pix_num, attrib, value))
#define CGLDescribePixelFormat PAL::softLink_OpenGL_CGLDescribePixelFormat
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, OpenGL, CGLDescribeRenderer, CGLError, (CGLRendererInfoObj rend, GLint rend_num, CGLRendererProperty prop, GLint *value), (rend, rend_num, prop, value))
#define CGLDescribeRenderer PAL::softLink_OpenGL_CGLDescribeRenderer
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, OpenGL, CGLDestroyContext, CGLError, (CGLContextObj ctx), (ctx))
#define CGLDestroyContext PAL::softLink_OpenGL_CGLDestroyContext
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, OpenGL, CGLDestroyRendererInfo, CGLError, (CGLRendererInfoObj rend), (rend))
#define CGLDestroyRendererInfo PAL::softLink_OpenGL_CGLDestroyRendererInfo
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, OpenGL, CGLGetParameter, CGLError, (CGLContextObj ctx, CGLContextParameter pname, GLint *params), (ctx, pname, params))
#define CGLGetParameter PAL::softLink_OpenGL_CGLGetParameter
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, OpenGL, CGLQueryRendererInfo, CGLError, (GLuint display_mask, CGLRendererInfoObj *rend, GLint *nrend), (display_mask, rend, nrend))
#define CGLQueryRendererInfo PAL::softLink_OpenGL_CGLQueryRendererInfo
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, OpenGL, CGLReleasePixelFormat, void, (CGLPixelFormatObj pix), (pix))
#define CGLReleasePixelFormat PAL::softLink_OpenGL_CGLReleasePixelFormat
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, OpenGL, CGLSetVirtualScreen, CGLError, (CGLContextObj ctx, GLint screen), (ctx, screen))
#define CGLSetVirtualScreen PAL::softLink_OpenGL_CGLSetVirtualScreen
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, OpenGL, CGLUpdateContext, CGLError, (CGLContextObj ctx), (ctx))
#define CGLUpdateContext PAL::softLink_OpenGL_CGLUpdateContext

#endif
