/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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
#include "WebGLObject.h"
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace JSC {
class AbstractSlotVisitor;
}

namespace WTF {
class AbstractLocker;
}

namespace WebCore {

class WebGL2RenderingContext;
class WebGLBuffer;
class WebGLProgram;

class WebGLTransformFeedback final : public WebGLObject {
public:
    virtual ~WebGLTransformFeedback();

    static RefPtr<WebGLTransformFeedback> create(WebGL2RenderingContext&);

    bool isActive() const { return m_active; }
    bool isPaused() const { return m_paused; }

    void setActive(bool active) { m_active = active; }
    void setPaused(bool paused) { m_paused = paused; }

    // These are the indexed bind points for transform feedback buffers.
    // Returns false if index is out of range and the caller should
    // synthesize a GL error.
    void setBoundIndexedTransformFeedbackBuffer(const AbstractLocker&, GCGLuint index, WebGLBuffer*);
    bool getBoundIndexedTransformFeedbackBuffer(GCGLuint index, WebGLBuffer** outBuffer);
    bool hasBoundIndexedTransformFeedbackBuffer(const WebGLBuffer* buffer) { return m_boundIndexedTransformFeedbackBuffers.contains(buffer); }

    bool validateProgramForResume(WebGLProgram*) const;

    void didBind() { m_hasEverBeenBound = true; }

    WebGLProgram* program() const { return m_program.get(); }
    void setProgram(const AbstractLocker&, WebGLProgram&);

    void unbindBuffer(const AbstractLocker&, WebGLBuffer&);

    bool hasEnoughBuffers(GCGLuint numRequired) const;

    void addMembersToOpaqueRoots(const AbstractLocker&, JSC::AbstractSlotVisitor&);

    bool isUsable() const { return object() && !isDeleted(); }
    bool isInitialized() const { return m_hasEverBeenBound; }

private:
    WebGLTransformFeedback(WebGL2RenderingContext&, PlatformGLObject);

    void deleteObjectImpl(const AbstractLocker&, GraphicsContextGL*, PlatformGLObject) override;

    bool m_active { false };
    bool m_paused { false };
    bool m_hasEverBeenBound { false };
    unsigned m_programLinkCount { 0 };
    Vector<WebGLBindingPoint<WebGLBuffer, GraphicsContextGL::TRANSFORM_FEEDBACK_BUFFER>> m_boundIndexedTransformFeedbackBuffers;
    RefPtr<WebGLProgram> m_program;
};

} // namespace WebCore

#endif // ENABLE(WEBGL)
