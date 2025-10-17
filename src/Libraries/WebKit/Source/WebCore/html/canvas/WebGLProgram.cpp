/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
#include "WebGLProgram.h"

#if ENABLE(WEBGL)

#include "InspectorInstrumentation.h"
#include "ScriptExecutionContext.h"
#include "WebCoreOpaqueRootInlines.h"
#include "WebGLRenderingContextBase.h"
#include "WebGLShader.h"
#include <JavaScriptCore/SlotVisitor.h>
#include <JavaScriptCore/SlotVisitorInlines.h>
#include <wtf/Lock.h>
#include <wtf/Locker.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

Lock WebGLProgram::s_instancesLock;

UncheckedKeyHashMap<WebGLProgram*, WebGLRenderingContextBase*>& WebGLProgram::instances()
{
    static NeverDestroyed<UncheckedKeyHashMap<WebGLProgram*, WebGLRenderingContextBase*>> instances;
    return instances;
}

Lock& WebGLProgram::instancesLock()
{
    return s_instancesLock;
}

RefPtr<WebGLProgram> WebGLProgram::create(WebGLRenderingContextBase& context)
{
    auto object = context.protectedGraphicsContextGL()->createProgram();
    if (!object)
        return nullptr;
    return adoptRef(*new WebGLProgram { context, object });
}

WebGLProgram::WebGLProgram(WebGLRenderingContextBase& context, PlatformGLObject object)
    : WebGLObject(context, object)
    , ContextDestructionObserver(context.scriptExecutionContext())
{
    ASSERT(scriptExecutionContext());

    {
        Locker locker { instancesLock() };
        instances().add(this, &context);
    }
}

WebGLProgram::~WebGLProgram()
{
    InspectorInstrumentation::willDestroyWebGLProgram(*this);

    {
        Locker locker { instancesLock() };
        ASSERT(instances().contains(this));
        instances().remove(this);
    }

    if (!m_context)
        return;

    runDestructor();
}

void WebGLProgram::contextDestroyed()
{
    InspectorInstrumentation::willDestroyWebGLProgram(*this);

    ContextDestructionObserver::contextDestroyed();
}

void WebGLProgram::deleteObjectImpl(const AbstractLocker& locker, GraphicsContextGL* context3d, PlatformGLObject obj)
{
    context3d->deleteProgram(obj);
    if (m_vertexShader) {
        m_vertexShader->onDetached(locker, context3d);
        m_vertexShader = nullptr;
    }
    if (m_fragmentShader) {
        m_fragmentShader->onDetached(locker, context3d);
        m_fragmentShader = nullptr;
    }
}

bool WebGLProgram::getLinkStatus()
{
    cacheInfoIfNeeded();
    return m_linkStatus;
}

void WebGLProgram::increaseLinkCount()
{
    ++m_linkCount;
    m_infoValid = false;
}

WebGLShader* WebGLProgram::getAttachedShader(GCGLenum type)
{
    switch (type) {
    case GraphicsContextGL::VERTEX_SHADER:
        return m_vertexShader.get();
    case GraphicsContextGL::FRAGMENT_SHADER:
        return m_fragmentShader.get();
    default:
        return 0;
    }
}

bool WebGLProgram::attachShader(const AbstractLocker&, WebGLShader* shader)
{
    if (!shader || !shader->object())
        return false;
    switch (shader->getType()) {
    case GraphicsContextGL::VERTEX_SHADER:
        if (m_vertexShader)
            return false;
        m_vertexShader = shader;
        return true;
    case GraphicsContextGL::FRAGMENT_SHADER:
        if (m_fragmentShader)
            return false;
        m_fragmentShader = shader;
        return true;
    default:
        return false;
    }
}

bool WebGLProgram::detachShader(const AbstractLocker&, WebGLShader* shader)
{
    if (!shader || !shader->object())
        return false;
    switch (shader->getType()) {
    case GraphicsContextGL::VERTEX_SHADER:
        if (m_vertexShader != shader)
            return false;
        m_vertexShader = nullptr;
        return true;
    case GraphicsContextGL::FRAGMENT_SHADER:
        if (m_fragmentShader != shader)
            return false;
        m_fragmentShader = nullptr;
        return true;
    default:
        return false;
    }
}

void WebGLProgram::addMembersToOpaqueRoots(const AbstractLocker&, JSC::AbstractSlotVisitor& visitor)
{
    addWebCoreOpaqueRoot(visitor, m_vertexShader.get());
    addWebCoreOpaqueRoot(visitor, m_fragmentShader.get());
}

void WebGLProgram::cacheInfoIfNeeded()
{
    if (m_infoValid)
        return;

    if (!object())
        return;

    RefPtr context = graphicsContextGL();
    if (!context)
        return;
    GCGLint linkStatus = context->getProgrami(object(), GraphicsContextGL::LINK_STATUS);
    m_linkStatus = linkStatus;
    if (m_linkStatus) {
        m_requiredTransformFeedbackBufferCount = m_requiredTransformFeedbackBufferCountAfterNextLink;
    }
    m_infoValid = true;
}

}

#endif // ENABLE(WEBGL)
