/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 12, 2023.
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
#include "WebGLCompressedTextureS3TCsRGB.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebGLCompressedTextureS3TCsRGB);

WebGLCompressedTextureS3TCsRGB::WebGLCompressedTextureS3TCsRGB(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::WebGLCompressedTextureS3TCsRGB)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_EXT_texture_compression_s3tc_srgb"_s);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB_S3TC_DXT1_EXT);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT);
}

WebGLCompressedTextureS3TCsRGB::~WebGLCompressedTextureS3TCsRGB() = default;

bool WebGLCompressedTextureS3TCsRGB::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_EXT_texture_compression_s3tc_srgb"_s);
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
