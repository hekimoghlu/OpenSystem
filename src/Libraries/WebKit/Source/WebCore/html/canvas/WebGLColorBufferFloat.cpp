/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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

#if ENABLE(WEBGL)
#include "WebGLColorBufferFloat.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebGLColorBufferFloat);

WebGLColorBufferFloat::WebGLColorBufferFloat(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::WebGLColorBufferFloat)
{
    RefPtr graphicsContextGL = context.graphicsContextGL();
    graphicsContextGL->ensureExtensionEnabled("GL_CHROMIUM_color_buffer_float_rgba"_s);
    // Optimistically enable RGB floating-point render targets also, if possible.
    graphicsContextGL->ensureExtensionEnabled("GL_CHROMIUM_color_buffer_float_rgb"_s);

    // https://github.com/KhronosGroup/WebGL/pull/2830
    // Spec requires EXT_float_blend to be turned on implicitly here.
    // Enable it both in the backend and in WebKit.
    context.getExtension("EXT_float_blend"_s);
}

WebGLColorBufferFloat::~WebGLColorBufferFloat() = default;

bool WebGLColorBufferFloat::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_OES_texture_float"_s)
        && context.supportsExtension("GL_CHROMIUM_color_buffer_float_rgba"_s);
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
