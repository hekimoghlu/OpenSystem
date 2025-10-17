/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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
#include "WebGLObject.h"

#if ENABLE(WEBGL)

#include "WebCoreOpaqueRoot.h"
#include "WebGLCompressedTextureS3TC.h"
#include "WebGLDebugRendererInfo.h"
#include "WebGLDebugShaders.h"
#include "WebGLLoseContext.h"
#include "WebGLRenderingContextBase.h"

namespace WebCore {

WebGLObject::WebGLObject(WebGLRenderingContextBase& context, PlatformGLObject object)
    : m_context(context.createRefForContextObject())
    , m_object(object)
{
}

WebGLObject::~WebGLObject() = default;

WebGLRenderingContextBase* WebGLObject::context() const
{
    return m_context.get();
}

Lock& WebGLObject::objectGraphLockForContext()
{
    // Should not call this if the object or context has been deleted.
    ASSERT(m_context);
    return m_context->objectGraphLock();
}

GraphicsContextGL* WebGLObject::graphicsContextGL() const
{
    return m_context ? m_context->graphicsContextGL() : nullptr;
}

void WebGLObject::runDestructor()
{
    auto& lock = objectGraphLockForContext();
    if (lock.isHeld()) {
        // Destruction of WebGLObjects can happen in chains triggered from GC.
        // The lock must be held only once, at the beginning of the chain.
        auto locker = AbstractLocker(NoLockingNecessary);
        deleteObject(locker, nullptr);
    } else {
        Locker locker { lock };
        deleteObject(locker, nullptr);
    }
}

void WebGLObject::deleteObject(const AbstractLocker& locker, GraphicsContextGL* context3d)
{
    m_deleted = true;
    if (!m_object)
        return;

    if (!m_context)
        return;

    if (!m_attachmentCount) {
        if (!context3d)
            context3d = graphicsContextGL();

        if (context3d)
            deleteObjectImpl(locker, context3d, m_object);
    }

    if (!m_attachmentCount)
        m_object = 0;
}

void WebGLObject::onDetached(const AbstractLocker& locker, GraphicsContextGL* context3d)
{
    ASSERT(m_attachmentCount); // FIXME: handle attachment with WebGLAttachmentPoint RAII object and remove the ifs.
    if (m_attachmentCount)
        --m_attachmentCount;
    if (m_deleted)
        deleteObject(locker, context3d);
}

bool WebGLObject::validate(const WebGLRenderingContextBase& context) const
{
    return &context == m_context;
}

WebCoreOpaqueRoot root(WebGLObject* object)
{
    return WebCoreOpaqueRoot { object };
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
