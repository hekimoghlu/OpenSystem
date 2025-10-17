/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#include "EXTTextureCompressionRGTC.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(EXTTextureCompressionRGTC);

EXTTextureCompressionRGTC::EXTTextureCompressionRGTC(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::EXTTextureCompressionRGTC)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_EXT_texture_compression_rgtc"_s);

    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RED_RGTC1_EXT);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SIGNED_RED_RGTC1_EXT);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_RED_GREEN_RGTC2_EXT);
    context.addCompressedTextureFormat(GraphicsContextGL::COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT);
}

EXTTextureCompressionRGTC::~EXTTextureCompressionRGTC() = default;

bool EXTTextureCompressionRGTC::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_EXT_texture_compression_rgtc"_s);
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
