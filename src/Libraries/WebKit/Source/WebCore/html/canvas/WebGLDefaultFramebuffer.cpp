/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#include "WebGLDefaultFramebuffer.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebGLDefaultFramebuffer);

std::unique_ptr<WebGLDefaultFramebuffer> WebGLDefaultFramebuffer::create(WebGLRenderingContextBase& context, IntSize size)
{
    auto instance = std::unique_ptr<WebGLDefaultFramebuffer> { new WebGLDefaultFramebuffer(context) };
    instance->reshape(size);
    return instance;
}

WebGLDefaultFramebuffer::WebGLDefaultFramebuffer(WebGLRenderingContextBase& context)
    : m_context(context)
{
    auto attributes = context.protectedGraphicsContextGL()->contextAttributes();
    m_hasStencil = attributes.stencil;
    m_hasDepth = attributes.depth;
    if (!attributes.preserveDrawingBuffer) {
        m_unpreservedBuffers = GraphicsContextGL::COLOR_BUFFER_BIT;
        if (m_hasStencil)
            m_unpreservedBuffers |= GraphicsContextGL::STENCIL_BUFFER_BIT;
        if (m_hasDepth)
            m_unpreservedBuffers |= GraphicsContextGL::DEPTH_BUFFER_BIT;
    }
}

IntSize WebGLDefaultFramebuffer::size() const
{
    return m_context->protectedGraphicsContextGL()->getInternalFramebufferSize();
}

void WebGLDefaultFramebuffer::reshape(IntSize size)
{
    m_context->protectedGraphicsContextGL()->reshape(size.width(), size.height());
}

void WebGLDefaultFramebuffer::markBuffersClear(GCGLbitfield clearBuffers)
{
    m_dirtyBuffers &= ~clearBuffers;
}

void WebGLDefaultFramebuffer::markAllUnpreservedBuffersDirty()
{
    m_dirtyBuffers = m_unpreservedBuffers;
}

void WebGLDefaultFramebuffer::markAllBuffersDirty()
{
    m_dirtyBuffers |= GraphicsContextGL::COLOR_BUFFER_BIT;
    if (m_hasStencil)
        m_dirtyBuffers |= GraphicsContextGL::STENCIL_BUFFER_BIT;
    if (m_hasDepth)
        m_dirtyBuffers |= GraphicsContextGL::DEPTH_BUFFER_BIT;
}

}

#endif
