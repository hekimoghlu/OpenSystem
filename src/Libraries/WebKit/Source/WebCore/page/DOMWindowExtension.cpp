/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
#include "DOMWindowExtension.h"

#include "DOMWrapperWorld.h"
#include "Document.h"
#include "FrameLoader.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "LocalFrameLoaderClient.h"
#include <wtf/Ref.h>

namespace WebCore {

DOMWindowExtension::DOMWindowExtension(LocalDOMWindow* window, DOMWrapperWorld& world)
    : m_window(window)
    , m_world(world)
    , m_wasDetached(false)
{
    ASSERT(this->frame());
    if (RefPtr window = m_window.get())
        window->registerObserver(*this);
}

DOMWindowExtension::~DOMWindowExtension()
{
    if (RefPtr window = m_window.get())
        window->unregisterObserver(*this);
}

LocalFrame* DOMWindowExtension::frame() const
{
    return m_window ? m_window->frame() : nullptr;
}

RefPtr<LocalFrame> DOMWindowExtension::protectedFrame() const
{
    return frame();
}

void DOMWindowExtension::suspendForBackForwardCache()
{
    // Calling out to the client might result in this DOMWindowExtension being destroyed
    // while there is still work to do.
    Ref protectedThis { *this };

    Ref frame = *this->frame();
    frame->protectedLoader()->client().dispatchWillDisconnectDOMWindowExtensionFromGlobalObject(this);

    m_disconnectedFrame = frame.get();
}

void DOMWindowExtension::resumeFromBackForwardCache()
{
    ASSERT(frame());
    ASSERT(m_disconnectedFrame == frame());
    ASSERT(frame()->document()->domWindow() == m_window);

    m_disconnectedFrame = nullptr;

    protectedFrame()->protectedLoader()->client().dispatchDidReconnectDOMWindowExtensionToGlobalObject(this);
}

void DOMWindowExtension::willDestroyGlobalObjectInCachedFrame()
{
    ASSERT(m_disconnectedFrame); // Somehow m_disconnectedFrame can be null here. See <rdar://problem/49613448>.

    // Calling out to the client might result in this DOMWindowExtension being destroyed
    // while there is still work to do.
    Ref protectedThis { *this };

    if (RefPtr disconnectedFrame = m_disconnectedFrame.get())
        disconnectedFrame->protectedLoader()->client().dispatchWillDestroyGlobalObjectForDOMWindowExtension(this);
    m_disconnectedFrame = nullptr;

    // DOMWindowExtension lifetime isn't tied directly to the LocalDOMWindow itself so it is important that it unregister
    // itself from any LocalDOMWindow it is associated with if that LocalDOMWindow is going away.
    ASSERT(m_window);
    if (RefPtr window = m_window.get())
        window->unregisterObserver(*this);
    m_window = nullptr;
}

void DOMWindowExtension::willDestroyGlobalObjectInFrame()
{
    ASSERT(!m_disconnectedFrame);

    // Calling out to the client might result in this DOMWindowExtension being destroyed
    // while there is still work to do.
    Ref protectedThis { *this };

    if (!m_wasDetached) {
        RefPtr frame = this->frame();
        ASSERT(frame);
        frame->protectedLoader()->client().dispatchWillDestroyGlobalObjectForDOMWindowExtension(this);
    }

    // DOMWindowExtension lifetime isn't tied directly to the LocalDOMWindow itself so it is important that it unregister
    // itself from any LocalDOMWindow it is associated with if that LocalDOMWindow is going away.
    ASSERT(m_window);
    if (RefPtr window = m_window.get())
        window->unregisterObserver(*this);
    m_window = nullptr;
}

void DOMWindowExtension::willDetachGlobalObjectFromFrame()
{
    ASSERT(!m_disconnectedFrame);
    ASSERT(!m_wasDetached);

    // Calling out to the client might result in this DOMWindowExtension being destroyed
    // while there is still work to do.
    Ref protectedThis { *this };

    RefPtr frame = this->frame();
    ASSERT(frame);
    frame->protectedLoader()->client().dispatchWillDestroyGlobalObjectForDOMWindowExtension(this);

    m_wasDetached = true;
}

} // namespace WebCore
