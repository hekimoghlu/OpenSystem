/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 12, 2023.
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
#include "WebGLVertexArrayObjectBase.h"

#if ENABLE(WEBGL)

#include "WebCoreOpaqueRootInlines.h"
#include "WebGLRenderingContextBase.h"
#include <JavaScriptCore/AbstractSlotVisitorInlines.h>
#include <wtf/Locker.h>

namespace WebCore {

WebGLVertexArrayObjectBase::WebGLVertexArrayObjectBase(WebGLRenderingContextBase& context, PlatformGLObject object, Type type)
    : WebGLObject(context, object)
    , m_type(type)
{
    m_vertexAttribState.grow(context.maxVertexAttribs());
}

void WebGLVertexArrayObjectBase::setElementArrayBuffer(const AbstractLocker& locker, WebGLBuffer* buffer)
{
    if (buffer)
        buffer->onAttached();
    if (m_boundElementArrayBuffer)
        m_boundElementArrayBuffer->onDetached(locker, context()->protectedGraphicsContextGL().get());
    m_boundElementArrayBuffer = buffer;
    
}
void WebGLVertexArrayObjectBase::setVertexAttribEnabled(int index, bool flag)
{
    auto& state = m_vertexAttribState[index];
    if (state.enabled == flag)
        return;
    state.enabled = flag;
    if (!state.validateBinding())
        m_allEnabledAttribBuffersBoundCache = false;
    else
        m_allEnabledAttribBuffersBoundCache.reset();
}

void WebGLVertexArrayObjectBase::setVertexAttribState(const AbstractLocker& locker, GCGLuint index, GCGLsizei bytesPerElement, GCGLint size, GCGLenum type, GCGLboolean normalized, GCGLsizei stride, GCGLintptr offset, bool isInteger, WebGLBuffer* buffer)
{
    auto& state = m_vertexAttribState[index];
    bool bindingWasValid = state.validateBinding();
    if (buffer)
        buffer->onAttached();
    if (state.bufferBinding)
        state.bufferBinding->onDetached(locker, context()->protectedGraphicsContextGL().get());
    state.bufferBinding = buffer;
    if (!state.validateBinding())
        m_allEnabledAttribBuffersBoundCache = false;
    else if (!bindingWasValid)
        m_allEnabledAttribBuffersBoundCache.reset();
    state.bytesPerElement = bytesPerElement;
    state.size = size;
    state.type = type;
    state.normalized = normalized;
    state.stride = stride ? stride : bytesPerElement;
    state.originalStride = stride;
    state.offset = offset;
    state.isInteger = isInteger;
}

void WebGLVertexArrayObjectBase::unbindBuffer(const AbstractLocker& locker, WebGLBuffer& buffer)
{
    if (m_boundElementArrayBuffer == &buffer) {
        m_boundElementArrayBuffer->onDetached(locker, context()->protectedGraphicsContextGL().get());
        m_boundElementArrayBuffer = nullptr;
    }
    
    for (auto& state : m_vertexAttribState) {
        if (state.bufferBinding == &buffer) {
            buffer.onDetached(locker, context()->protectedGraphicsContextGL().get());
            state.bufferBinding = nullptr;
            if (!state.validateBinding())
                m_allEnabledAttribBuffersBoundCache = false;
        }
    }
}

void WebGLVertexArrayObjectBase::setVertexAttribDivisor(GCGLuint index, GCGLuint divisor)
{
    m_vertexAttribState[index].divisor = divisor;
}

void WebGLVertexArrayObjectBase::addMembersToOpaqueRoots(const AbstractLocker&, JSC::AbstractSlotVisitor& visitor)
{
    addWebCoreOpaqueRoot(visitor, m_boundElementArrayBuffer.get());
    for (auto& state : m_vertexAttribState)
        addWebCoreOpaqueRoot(visitor, state.bufferBinding.get());
}

bool WebGLVertexArrayObjectBase::areAllEnabledAttribBuffersBound()
{
    if (!m_allEnabledAttribBuffersBoundCache) {
        m_allEnabledAttribBuffersBoundCache = [&] {
            for (const auto& state : m_vertexAttribState) {
                if (!state.validateBinding())
                    return false;
            }
            return true;
        }();
    }
    return *m_allEnabledAttribBuffersBoundCache;
}

WebCoreOpaqueRoot root(WebGLVertexArrayObjectBase* array)
{
    return WebCoreOpaqueRoot { array };
}

}

#endif // ENABLE(WEBGL)
