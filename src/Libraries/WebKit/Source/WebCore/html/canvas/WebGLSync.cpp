/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#include "WebGLSync.h"

#include "HTMLCanvasElement.h"
#include "WebGLRenderingContextBase.h"
#include <wtf/Lock.h>
#include <wtf/Locker.h>

namespace WebCore {

RefPtr<WebGLSync> WebGLSync::create(WebGLRenderingContextBase& context)
{
    auto object = context.protectedGraphicsContextGL()->fenceSync(GraphicsContextGL::SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!object)
        return nullptr;
    return adoptRef(*new WebGLSync { context, object });
}

WebGLSync::~WebGLSync()
{
    if (!m_context)
        return;

    runDestructor();
}

WebGLSync::WebGLSync(WebGLRenderingContextBase& context, GCGLsync object)
    : WebGLObject(context, static_cast<PlatformGLObject>(-1)) // This value is unused because the sync object is a pointer type, but it needs to be non-zero or other parts of the code will assume the object is invalid.
    , m_sync(object)
{
}

void WebGLSync::deleteObjectImpl(const AbstractLocker&, GraphicsContextGL* context3d, PlatformGLObject)
{
    context3d->deleteSync(m_sync);
    m_sync = nullptr;
}

void WebGLSync::updateCache(WebGLRenderingContextBase& context)
{
    if (m_syncStatus == GraphicsContextGL::SIGNALED || !m_allowCacheUpdate)
        return;

    m_allowCacheUpdate = false;
    m_syncStatus = context.protectedGraphicsContextGL()->getSynci(m_sync, GraphicsContextGL::SYNC_STATUS);
    if (m_syncStatus == GraphicsContextGL::UNSIGNALED)
        scheduleAllowCacheUpdate(context);
}

GCGLint WebGLSync::getCachedResult(GCGLenum pname) const
{
    switch (pname) {
    case GraphicsContextGL::OBJECT_TYPE:
        return GraphicsContextGL::SYNC_FENCE;
    case GraphicsContextGL::SYNC_STATUS:
        return m_syncStatus;
    case GraphicsContextGL::SYNC_CONDITION:
        return GraphicsContextGL::SYNC_GPU_COMMANDS_COMPLETE;
    case GraphicsContextGL::SYNC_FLAGS:
        return 0;
    }
    ASSERT_NOT_REACHED();
    return 0;
}

bool WebGLSync::isSignaled() const
{
    return m_syncStatus == GraphicsContextGL::SIGNALED;
}

void WebGLSync::scheduleAllowCacheUpdate(WebGLRenderingContextBase& context)
{
    context.canvasBase().queueTaskKeepingObjectAlive(TaskSource::WebGL, [protectedThis = Ref { *this }]() {
        protectedThis->m_allowCacheUpdate = true;
    });
}

}

#endif // ENABLE(WEBGL)
