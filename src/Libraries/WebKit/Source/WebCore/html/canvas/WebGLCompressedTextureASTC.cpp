/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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
#include "WebGLCompressedTextureASTC.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebGLCompressedTextureASTC);

WebGLCompressedTextureASTC::WebGLCompressedTextureASTC(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::WebGLCompressedTextureASTC)
    , m_isHDRSupported(context.protectedGraphicsContextGL()->supportsExtension("GL_KHR_texture_compression_astc_hdr"_s))
    , m_isLDRSupported(context.protectedGraphicsContextGL()->supportsExtension("GL_KHR_texture_compression_astc_ldr"_s))
{
    RefPtr graphicsContextGL = context.graphicsContextGL();
    graphicsContextGL->ensureExtensionEnabled("GL_KHR_texture_compression_astc_hdr"_s);
    graphicsContextGL->ensureExtensionEnabled("GL_KHR_texture_compression_astc_ldr"_s);

    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_4x4_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_5x4_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_5x5_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_6x5_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_6x6_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_8x5_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_8x6_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_8x8_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_10x5_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_10x6_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_10x8_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_10x10_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_12x10_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_ASTC_12x12_KHR);
    
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR);
}

WebGLCompressedTextureASTC::~WebGLCompressedTextureASTC() = default;
    
Vector<String> WebGLCompressedTextureASTC::getSupportedProfiles()
{
    Vector<String> result;
    
    if (m_isHDRSupported)
        result.append("hdr"_s);
    if (m_isLDRSupported)
        result.append("ldr"_s);
    
    return result;
}

bool WebGLCompressedTextureASTC::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_KHR_texture_compression_astc_hdr"_s)
        || context.supportsExtension("GL_KHR_texture_compression_astc_ldr"_s);
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
