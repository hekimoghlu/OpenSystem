/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
#include "WebGLCompressedTextureS3TC.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebGLCompressedTextureS3TC);

WebGLCompressedTextureS3TC::WebGLCompressedTextureS3TC(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::WebGLCompressedTextureS3TC)
{
    RefPtr gcgl = context.graphicsContextGL();
    gcgl->ensureExtensionEnabled("GL_EXT_texture_compression_dxt1"_s);
    gcgl->ensureExtensionEnabled("GL_ANGLE_texture_compression_dxt3"_s);
    gcgl->ensureExtensionEnabled("GL_ANGLE_texture_compression_dxt5"_s);

    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGB_S3TC_DXT1_EXT);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_S3TC_DXT1_EXT);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_S3TC_DXT3_EXT);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_S3TC_DXT5_EXT);
}

WebGLCompressedTextureS3TC::~WebGLCompressedTextureS3TC() = default;

bool WebGLCompressedTextureS3TC::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_EXT_texture_compression_dxt1"_s)
        && context.supportsExtension("GL_ANGLE_texture_compression_dxt3"_s)
        && context.supportsExtension("GL_ANGLE_texture_compression_dxt5"_s);
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
