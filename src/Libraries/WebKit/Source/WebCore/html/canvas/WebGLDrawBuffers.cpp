/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
#include "WebGLDrawBuffers.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebGLDrawBuffers);

WebGLDrawBuffers::WebGLDrawBuffers(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::WebGLDrawBuffers)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_EXT_draw_buffers"_s);
}

WebGLDrawBuffers::~WebGLDrawBuffers() = default;

bool WebGLDrawBuffers::supported(WebGLRenderingContextBase& context)
{
    return context.protectedGraphicsContextGL()->supportsExtension("GL_EXT_draw_buffers"_s);
}

void WebGLDrawBuffers::drawBuffersWEBGL(const Vector<GCGLenum>& buffers)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    GCGLsizei n = buffers.size();
    auto bufs = buffers.span();
    if (!context.m_framebufferBinding) {
        if (n != 1) {
            context.synthesizeGLError(GraphicsContextGL::INVALID_OPERATION, "drawBuffersWEBGL"_s, "more or fewer than one buffer"_s);
            return;
        }
        if (bufs[0] != GraphicsContextGL::BACK && bufs[0] != GraphicsContextGL::NONE) {
            context.synthesizeGLError(GraphicsContextGL::INVALID_OPERATION, "drawBuffersWEBGL"_s, "BACK or NONE"_s);
            return;
        }
        // Because the backbuffer is simulated on all current WebKit ports, we need to change BACK to COLOR_ATTACHMENT0.
        GCGLenum value[1] { bufs[0] == GraphicsContextGL::BACK ? GraphicsContextGL::COLOR_ATTACHMENT0 : GraphicsContextGL::NONE };
        context.protectedGraphicsContextGL()->drawBuffersEXT(value);
        context.setBackDrawBuffer(bufs[0]);
    } else {
        if (n > context.maxDrawBuffers()) {
            context.synthesizeGLError(GraphicsContextGL::INVALID_VALUE, "drawBuffersWEBGL"_s, "more than max draw buffers"_s);
            return;
        }
        for (GCGLsizei i = 0; i < n; ++i) {
            if (bufs[i] != GraphicsContextGL::NONE && bufs[i] != GraphicsContextGL::COLOR_ATTACHMENT0_EXT + i) {
                context.synthesizeGLError(GraphicsContextGL::INVALID_OPERATION, "drawBuffersWEBGL"_s, "COLOR_ATTACHMENTi_EXT or NONE"_s);
                return;
            }
        }
        context.m_framebufferBinding->drawBuffers(buffers);
    }
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
