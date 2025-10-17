/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
#include "WebGLTransformFeedback.h"

#include "WebCoreOpaqueRootInlines.h"
#include "WebGL2RenderingContext.h"
#include "WebGLBuffer.h"
#include <JavaScriptCore/AbstractSlotVisitorInlines.h>
#include <wtf/Lock.h>
#include <wtf/Locker.h>

namespace WebCore {

RefPtr<WebGLTransformFeedback> WebGLTransformFeedback::create(WebGL2RenderingContext& context)
{
    auto object = context.protectedGraphicsContextGL()->createTransformFeedback();
    if (!object)
        return nullptr;
    return adoptRef(*new WebGLTransformFeedback { context, object });
}

WebGLTransformFeedback::~WebGLTransformFeedback()
{
    if (!m_context)
        return;

    runDestructor();
}

WebGLTransformFeedback::WebGLTransformFeedback(WebGL2RenderingContext& context, PlatformGLObject object)
    : WebGLObject(context, object)
{
    m_boundIndexedTransformFeedbackBuffers.grow(context.maxTransformFeedbackSeparateAttribs());
}

void WebGLTransformFeedback::deleteObjectImpl(const AbstractLocker&, GraphicsContextGL* context3d, PlatformGLObject object)
{
    context3d->deleteTransformFeedback(object);
}

void WebGLTransformFeedback::setProgram(const AbstractLocker&, WebGLProgram& program)
{
    m_program = &program;
    m_programLinkCount = program.getLinkCount();
}

void WebGLTransformFeedback::setBoundIndexedTransformFeedbackBuffer(const AbstractLocker&, GCGLuint index, WebGLBuffer* buffer)
{
    ASSERT(index < m_boundIndexedTransformFeedbackBuffers.size());
    m_boundIndexedTransformFeedbackBuffers[index] = buffer;
}

bool WebGLTransformFeedback::getBoundIndexedTransformFeedbackBuffer(GCGLuint index, WebGLBuffer** outBuffer)
{
    if (index >= m_boundIndexedTransformFeedbackBuffers.size())
        return false;
    *outBuffer = m_boundIndexedTransformFeedbackBuffers[index].get();
    return true;
}

bool WebGLTransformFeedback::hasEnoughBuffers(GCGLuint numRequired) const
{
    if (numRequired > m_boundIndexedTransformFeedbackBuffers.size())
        return false;
    for (GCGLuint i = 0; i < numRequired; i++) {
        if (!m_boundIndexedTransformFeedbackBuffers[i].get())
            return false;
    }
    return true;
}

void WebGLTransformFeedback::addMembersToOpaqueRoots(const AbstractLocker& locker, JSC::AbstractSlotVisitor& visitor)
{
    for (auto& buffer : m_boundIndexedTransformFeedbackBuffers)
        addWebCoreOpaqueRoot(visitor, buffer.get());

    addWebCoreOpaqueRoot(visitor, m_program.get());
    if (m_program)
        m_program->addMembersToOpaqueRoots(locker, visitor);
}

void WebGLTransformFeedback::unbindBuffer(const AbstractLocker&, WebGLBuffer& buffer)
{
    for (auto& boundBuffer : m_boundIndexedTransformFeedbackBuffers) {
        if (boundBuffer == &buffer)
            boundBuffer = nullptr;
    }
}

bool WebGLTransformFeedback::validateProgramForResume(WebGLProgram* program) const
{
    return program && m_program == program && program->getLinkCount() == m_programLinkCount;
}

}

#endif // ENABLE(WEBGL)
