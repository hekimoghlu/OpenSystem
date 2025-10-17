/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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
#include "WebGLCompressedTexturePVRTC.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebGLCompressedTexturePVRTC);

WebGLCompressedTexturePVRTC::WebGLCompressedTexturePVRTC(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::WebGLCompressedTexturePVRTC)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_IMG_texture_compression_pvrtc"_s);

    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGB_PVRTC_4BPPV1_IMG);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGB_PVRTC_2BPPV1_IMG);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_PVRTC_4BPPV1_IMG);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RGBA_PVRTC_2BPPV1_IMG);
}

WebGLCompressedTexturePVRTC::~WebGLCompressedTexturePVRTC() = default;

bool WebGLCompressedTexturePVRTC::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_IMG_texture_compression_pvrtc"_s);
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
