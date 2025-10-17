/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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

#include "EventTarget.h"
#include "ExceptionOr.h"
#include "PlatformMouseEvent.h"
#include "PointerID.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Document;
class Element;
class EventTarget;
class IntPoint;
class MouseEvent;
class Page;
class PlatformTouchEvent;
class PointerEvent;
class WindowProxy;

class PointerCaptureController {
    WTF_MAKE_NONCOPYABLE(PointerCaptureController);
    WTF_MAKE_TZONE_ALLOCATED(PointerCaptureController);
public:
    explicit PointerCaptureController(Page&);

    Element* pointerCaptureElement(Document*, PointerID) const;
    ExceptionOr<void> setPointerCapture(Element*, PointerID);
    ExceptionOr<void> releasePointerCapture(Element*, PointerID);
    bool hasPointerCapture(Element*, PointerID);
    void reset();

    void pointerLockWasApplied();
    void elementWasRemoved(Element&);

    RefPtr<PointerEvent> pointerEventForMouseEvent(const MouseEvent&, PointerID, const String& pointerType);

#if ENABLE(TOUCH_EVENTS) && (PLATFORM(IOS_FAMILY) || PLATFORM(WPE))
    void dispatchEventForTouchAtIndex(EventTarget&, const PlatformTouchEvent&, unsigned, bool isPrimary, WindowProxy&, const IntPoint&);
#endif

    WEBCORE_EXPORT void touchWithIdentifierWasRemoved(PointerID);
    bool hasCancelledPointerEventForIdentifier(PointerID) const;
    bool preventsCompatibilityMouseEventsForIdentifier(PointerID) const;
    void dispatchEvent(PointerEvent&, EventTarget*);
    WEBCORE_EXPORT void cancelPointer(PointerID, const IntPoint&, PointerEvent* existingCancelEvent = nullptr);
    void processPendingPointerCapture(PointerID);

private:
    struct CapturingData : public RefCounted<CapturingData> {
        static Ref<CapturingData> create(const String& pointerType)
        {
            return adoptRef(*new CapturingData(pointerType));
        }

        WeakPtr<Document, WeakPtrImplWithEventTargetData> activeDocument;
        RefPtr<Element> pendingTargetOverride;
        RefPtr<Element> targetOverride;
#if ENABLE(TOUCH_EVENTS) && (PLATFORM(IOS_FAMILY) || PLATFORM(WPE))
        RefPtr<Element> previousTarget;
#endif
        bool hasAnyElement() const {
            return pendingTargetOverride || targetOverride
#if ENABLE(TOUCH_EVENTS) && (PLATFORM(IOS_FAMILY) || PLATFORM(WPE))
                || previousTarget
#endif
                ;
        }
        String pointerType;
        enum class State : uint8_t {
            Ready,
            Finished,
            Cancelled,
        };
        State state { State::Ready };
        bool isPrimary { false };
        bool preventsCompatibilityMouseEvents { false };
        bool pointerIsPressed { false };
        MouseButton previousMouseButton { MouseButton::PointerHasNotChanged };

    private:
        CapturingData(const String& pointerType)
            : pointerType(pointerType)
        { }
    };

    Ref<CapturingData> ensureCapturingDataForPointerEvent(const PointerEvent&);
    void pointerEventWillBeDispatched(const PointerEvent&, EventTarget*);
    void pointerEventWasDispatched(const PointerEvent&);

    void updateHaveAnyCapturingElement();
    void elementWasRemovedSlow(Element&);

    void dispatchOverOrOutEvent(const AtomString&, EventTarget*, const PlatformTouchEvent&, unsigned index, bool isPrimary, WindowProxy&, IntPoint);
    void dispatchEnterOrLeaveEvent(const AtomString&, Element&, const PlatformTouchEvent&, unsigned index, bool isPrimary, WindowProxy&, IntPoint);

    WeakPtr<Page> m_page;
    // While PointerID is defined as int32_t, we use int64_t here so that we may use a value outside of the int32_t range to have safe
    // empty and removed values, allowing any int32_t to be provided through the API for lookup in this hashmap.
    using PointerIdToCapturingDataMap = UncheckedKeyHashMap<int64_t, Ref<CapturingData>, IntHash<int64_t>, WTF::SignedWithZeroKeyHashTraits<int64_t>>;
    PointerIdToCapturingDataMap m_activePointerIdsToCapturingData;
    bool m_processingPendingPointerCapture { false };
    bool m_haveAnyCapturingElement { false };
};

inline void PointerCaptureController::elementWasRemoved(Element& element)
{
    if (m_haveAnyCapturingElement)
        elementWasRemovedSlow(element);
}

} // namespace WebCore
