/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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

#if ENABLE(POINTER_LOCK)

#include "ExceptionCode.h"
#include "PointerLockOptions.h"

#include <optional>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class Element;
class DeferredPromise;
class Document;
class Page;
class PlatformMouseEvent;
class PlatformWheelEvent;
class VoidCallback;
class WeakPtrImplWithEventTargetData;

class PointerLockController {
    WTF_MAKE_NONCOPYABLE(PointerLockController);
    WTF_MAKE_TZONE_ALLOCATED(PointerLockController);
public:
    explicit PointerLockController(Page&);
    ~PointerLockController();
    void requestPointerLock(Element* target, std::optional<PointerLockOptions>&& = std::nullopt, RefPtr<DeferredPromise> = nullptr);

    void requestPointerUnlock();
    void requestPointerUnlockAndForceCursorVisible();
    void elementWasRemoved(Element&);
    void documentDetached(Document&);
    bool isLocked() const;
    WEBCORE_EXPORT bool lockPending() const;
    WEBCORE_EXPORT Element* element() const;

    WEBCORE_EXPORT void didAcquirePointerLock();
    WEBCORE_EXPORT void didNotAcquirePointerLock();
    WEBCORE_EXPORT void didLosePointerLock();
    void dispatchLockedMouseEvent(const PlatformMouseEvent&, const AtomString& eventType);
    void dispatchLockedWheelEvent(const PlatformWheelEvent&);

    static bool supportsUnadjustedMovement();

private:
    void clearElement();
    void enqueueEvent(const AtomString& type, Element*);
    void enqueueEvent(const AtomString& type, Document*);
    void resolvePromises();
    void rejectPromises(ExceptionCode, const String&);
    void elementWasRemovedInternal();

    Page& m_page;
    bool m_lockPending { false };
    bool m_unlockPending { false };
    bool m_forceCursorVisibleUponUnlock { false };
    std::optional<PointerLockOptions> m_options;
    RefPtr<Element> m_element;
    Vector<Ref<DeferredPromise>> m_promises;
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_documentOfRemovedElementWhileWaitingForUnlock;
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_documentAllowedToRelockWithoutUserGesture;
};

inline void PointerLockController::elementWasRemoved(Element& element)
{
    if (m_element == &element)
        elementWasRemovedInternal();
}

} // namespace WebCore

#endif // ENABLE(POINTER_LOCK)
