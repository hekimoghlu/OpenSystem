/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
#include "config.h"

#if PLATFORM(MAC)

#include <OpenGL/OpenGL.h>
#include <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, OpenGL, PAL_EXPORT)

SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(PAL, OpenGL, CGLChoosePixelFormat, CGLError, (const CGLPixelFormatAttribute *attribs, CGLPixelFormatObj *pix, GLint *npix), (attribs, pix, npix), PAL_EXPORT)
SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(PAL, OpenGL, CGLReleasePixelFormat, void, (CGLPixelFormatObj pix), (pix), PAL_EXPORT)

SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, OpenGL, CGLCreateContext, CGLError, (CGLPixelFormatObj pix, CGLContextObj share, CGLContextObj *ctx), (pix, share, ctx))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, OpenGL, CGLDescribePixelFormat, CGLError, (CGLPixelFormatObj pix, GLint pix_num, CGLPixelFormatAttribute attrib, GLint *value), (pix, pix_num, attrib, value))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, OpenGL, CGLDescribeRenderer, CGLError, (CGLRendererInfoObj rend, GLint rend_num, CGLRendererProperty prop, GLint *value), (rend, rend_num, prop, value))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, OpenGL, CGLDestroyContext, CGLError, (CGLContextObj ctx), (ctx))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, OpenGL, CGLDestroyRendererInfo, CGLError, (CGLRendererInfoObj rend), (rend))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, OpenGL, CGLGetParameter, CGLError, (CGLContextObj ctx, CGLContextParameter pname, GLint *params), (ctx, pname, params))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, OpenGL, CGLQueryRendererInfo, CGLError, (GLuint display_mask, CGLRendererInfoObj *rend, GLint *nrend), (display_mask, rend, nrend))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, OpenGL, CGLSetVirtualScreen, CGLError, (CGLContextObj ctx, GLint screen), (ctx, screen))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, OpenGL, CGLUpdateContext, CGLError, (CGLContextObj ctx), (ctx))

#endif
