/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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
#include "OESTextureHalfFloat.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(OESTextureHalfFloat);

OESTextureHalfFloat::OESTextureHalfFloat(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::OESTextureHalfFloat)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_OES_texture_half_float"_s);

    // Spec requires EXT_color_buffer_half_float to be turned on implicitly here.
    // Enable it both in the backend and in WebKit.
    context.getExtension("EXT_color_buffer_half_float"_s);
}

OESTextureHalfFloat::~OESTextureHalfFloat() = default;

bool OESTextureHalfFloat::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_OES_texture_half_float"_s);
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
