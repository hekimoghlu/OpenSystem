/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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
#include "WebGLDebugShaders.h"

#include "WebGLShader.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebGLDebugShaders);

WebGLDebugShaders::WebGLDebugShaders(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::WebGLDebugShaders)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_ANGLE_translated_shader_source"_s);
}

WebGLDebugShaders::~WebGLDebugShaders() = default;

bool WebGLDebugShaders::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_ANGLE_translated_shader_source"_s);
}

String WebGLDebugShaders::getTranslatedShaderSource(WebGLShader& shader)
{
    if (isContextLost())
        return String();
    auto& context = this->context();
    if (!context.validateWebGLObject("getTranslatedShaderSource"_s, shader))
        return emptyString();
    return context.protectedGraphicsContextGL()->getTranslatedShaderSourceANGLE(shader.object());
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
