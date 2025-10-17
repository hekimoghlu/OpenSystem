/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#include "WebGLCompressedTextureETC.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebGLCompressedTextureETC);

WebGLCompressedTextureETC::WebGLCompressedTextureETC(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::WebGLCompressedTextureETC)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_ANGLE_compressed_texture_etc"_s);

    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_R11_EAC);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SIGNED_R11_EAC);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RG11_EAC);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SIGNED_RG11_EAC);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGB8_ETC2);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ETC2);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA8_ETC2_EAC);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ETC2_EAC);
}

WebGLCompressedTextureETC::~WebGLCompressedTextureETC() = default;

bool WebGLCompressedTextureETC::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_ANGLE_compressed_texture_etc"_s);
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
