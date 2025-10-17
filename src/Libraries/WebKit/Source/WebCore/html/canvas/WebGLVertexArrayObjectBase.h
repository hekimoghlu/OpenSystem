/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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
#pragma once

#if ENABLE(WEBGL)

#include "GraphicsContextGL.h"
#include "WebGLBuffer.h"
#include "WebGLObject.h"
#include <optional>

namespace JSC {
class AbstractSlotVisitor;
}

namespace WTF {
class AbstractLocker;
}

namespace WebCore {

class WebCoreOpaqueRoot;

class WebGLVertexArrayObjectBase : public WebGLObject {
public:
    enum class Type { Default, User };

    // Cached values for vertex attrib range checks
    struct VertexAttribState {
        bool isBound() const { return bufferBinding && bufferBinding->object(); }
        bool validateBinding() const { return !enabled || isBound(); }

        bool enabled { false };
        WebGLBindingPoint<WebGLBuffer, GraphicsContextGL::ARRAY_BUFFER> bufferBinding;
        GCGLsizei bytesPerElement { 0 };
        GCGLint size { 4 };
        GCGLenum type { GraphicsContextGL::FLOAT };
        bool normalized { false };
        GCGLsizei stride { 16 };
        GCGLsizei originalStride { 0 };
        GCGLintptr offset { 0 };
        GCGLuint divisor { 0 };
        bool isInteger { false };
    };

    bool isDefaultObject() const { return m_type == Type::Default; }

    void didBind() { m_hasEverBeenBound = true; }

    WebGLBuffer* getElementArrayBuffer() const { return m_boundElementArrayBuffer.get(); }
    void setElementArrayBuffer(const AbstractLocker&, WebGLBuffer*);

    void setVertexAttribEnabled(int index, bool flag);
    const VertexAttribState& getVertexAttribState(int index) { return m_vertexAttribState[index]; }
    void setVertexAttribState(const AbstractLocker&, GCGLuint, GCGLsizei, GCGLint, GCGLenum, GCGLboolean, GCGLsizei, GCGLintptr, bool, WebGLBuffer*);
    bool hasArrayBuffer(WebGLBuffer* buffer) { return m_vertexAttribState.containsIf([&](auto& item) { return item.bufferBinding == buffer; }); }
    void unbindBuffer(const AbstractLocker&, WebGLBuffer&);

    void setVertexAttribDivisor(GCGLuint index, GCGLuint divisor);

    void addMembersToOpaqueRoots(const AbstractLocker&, JSC::AbstractSlotVisitor&);

    bool areAllEnabledAttribBuffersBound();

    bool isUsable() const { return object() && !isDeleted(); }
    bool isInitialized() const { return m_hasEverBeenBound; }

protected:
    WebGLVertexArrayObjectBase(WebGLRenderingContextBase&, PlatformGLObject, Type);
    void deleteObjectImpl(const AbstractLocker&, GraphicsContextGL*, PlatformGLObject) override = 0;

    Type m_type;
    bool m_hasEverBeenBound { false };
    WebGLBindingPoint<WebGLBuffer, GraphicsContextGL::ELEMENT_ARRAY_BUFFER> m_boundElementArrayBuffer;
    Vector<VertexAttribState> m_vertexAttribState;
    std::optional<bool> m_allEnabledAttribBuffersBoundCache;
};

WebCoreOpaqueRoot root(WebGLVertexArrayObjectBase*);

} // namespace WebCore

#endif
