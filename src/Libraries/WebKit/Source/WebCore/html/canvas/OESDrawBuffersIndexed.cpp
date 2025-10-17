/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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
#include "OESDrawBuffersIndexed.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(OESDrawBuffersIndexed);

OESDrawBuffersIndexed::OESDrawBuffersIndexed(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::OESDrawBuffersIndexed)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_OES_draw_buffers_indexed"_s);
}

OESDrawBuffersIndexed::~OESDrawBuffersIndexed() = default;

bool OESDrawBuffersIndexed::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_OES_draw_buffers_indexed"_s);
}

void OESDrawBuffersIndexed::enableiOES(GCGLenum target, GCGLuint index)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    context.protectedGraphicsContextGL()->enableiOES(target, index);
}

void OESDrawBuffersIndexed::disableiOES(GCGLenum target, GCGLuint index)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    context.protectedGraphicsContextGL()->disableiOES(target, index);
}

void OESDrawBuffersIndexed::blendEquationiOES(GCGLuint buf, GCGLenum mode)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    context.protectedGraphicsContextGL()->blendEquationiOES(buf, mode);
}

void OESDrawBuffersIndexed::blendEquationSeparateiOES(GCGLuint buf, GCGLenum modeRGB, GCGLenum modeAlpha)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    context.protectedGraphicsContextGL()->blendEquationSeparateiOES(buf, modeRGB, modeAlpha);
}

void OESDrawBuffersIndexed::blendFunciOES(GCGLuint buf, GCGLenum src, GCGLenum dst)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    context.protectedGraphicsContextGL()->blendFunciOES(buf, src, dst);
}

void OESDrawBuffersIndexed::blendFuncSeparateiOES(GCGLuint buf, GCGLenum srcRGB, GCGLenum dstRGB, GCGLenum srcAlpha, GCGLenum dstAlpha)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    context.protectedGraphicsContextGL()->blendFuncSeparateiOES(buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
}

void OESDrawBuffersIndexed::colorMaskiOES(GCGLuint buf, GCGLboolean red, GCGLboolean green, GCGLboolean blue, GCGLboolean alpha)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    // Used in WebGLRenderingContextBase::clearIfComposited
    if (!buf) {
        context.m_colorMask[0] = red;
        context.m_colorMask[1] = green;
        context.m_colorMask[2] = blue;
        context.m_colorMask[3] = alpha;
    }
    context.protectedGraphicsContextGL()->colorMaskiOES(buf, red, green, blue, alpha);
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
