/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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

#include "LocalDOMWindow.h"
#include <wtf/CheckedRef.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DOMWrapperWorld;
class LocalFrame;

class DOMWindowExtension final : public RefCounted<DOMWindowExtension>, public LocalDOMWindowObserver {
public:
    static Ref<DOMWindowExtension> create(LocalDOMWindow* window, DOMWrapperWorld& world)
    {
        return adoptRef(*new DOMWindowExtension(window, world));
    }

    WEBCORE_EXPORT ~DOMWindowExtension();

    void suspendForBackForwardCache() final;
    void resumeFromBackForwardCache() final;
    void willDestroyGlobalObjectInCachedFrame() final;
    void willDestroyGlobalObjectInFrame() final;
    void willDetachGlobalObjectFromFrame() final;

    WEBCORE_EXPORT LocalFrame* frame() const;
    RefPtr<LocalFrame> protectedFrame() const;
    DOMWrapperWorld& world() const { return m_world; }

private:
    WEBCORE_EXPORT DOMWindowExtension(LocalDOMWindow*, DOMWrapperWorld&);

    WeakPtr<LocalDOMWindow, WeakPtrImplWithEventTargetData> m_window;
    Ref<DOMWrapperWorld> m_world;
    WeakPtr<LocalFrame> m_disconnectedFrame;
    bool m_wasDetached;
};

} // namespace WebCore
