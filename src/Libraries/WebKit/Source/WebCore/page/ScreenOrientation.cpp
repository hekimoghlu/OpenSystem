/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
#include "ScreenOrientation.h"

#include "Document.h"
#include "DocumentInlines.h"
#include "Element.h"
#include "Event.h"
#include "EventNames.h"
#include "FrameDestructionObserverInlines.h"
#include "FullscreenManager.h"
#include "JSDOMPromiseDeferred.h"
#include "LocalDOMWindow.h"
#include "Page.h"
#include "VisibilityState.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ScreenOrientation);

Ref<ScreenOrientation> ScreenOrientation::create(Document* document)
{
    auto screenOrientation = adoptRef(*new ScreenOrientation(document));
    screenOrientation->suspendIfNeeded();
    return screenOrientation;
}

ScreenOrientation::ScreenOrientation(Document* document)
    : ActiveDOMObject(document)
{
    if (shouldListenForChangeNotification()) {
        if (auto* manager = this->manager())
            manager->addObserver(*this);
    }
}

ScreenOrientation::~ScreenOrientation()
{
    if (auto* manager = this->manager())
        manager->removeObserver(*this);
}

Document* ScreenOrientation::document() const
{
    return downcast<Document>(scriptExecutionContext());
}

ScreenOrientationManager* ScreenOrientation::manager() const
{
    RefPtr document = this->document();
    if (!document)
        return nullptr;
    auto* page = document->page();
    return page ? page->screenOrientationManager() : nullptr;
}

static bool isSupportedLockType(ScreenOrientationLockType lockType)
{
    switch (lockType) {
    case ScreenOrientationLockType::Any:
    case ScreenOrientationLockType::Natural:
    case ScreenOrientationLockType::Portrait:
    case ScreenOrientationLockType::Landscape:
        return true;
    default:
        return false;
    }
}

void ScreenOrientation::lock(LockType lockType, Ref<DeferredPromise>&& promise)
{
    RefPtr document = this->document();
    if (!document || !document->isFullyActive()) {
        promise->reject(Exception { ExceptionCode::InvalidStateError, "Document is not fully active."_s });
        return;
    }

    auto* manager = this->manager();
    if (!manager) {
        promise->reject(Exception { ExceptionCode::InvalidStateError, "No browsing context"_s });
        return;
    }

    // FIXME: Add support for the sandboxed orientation lock browsing context flag.
    if (!document->isSameOriginAsTopDocument()) {
        promise->reject(Exception { ExceptionCode::SecurityError, "Only first party documents can lock the screen orientation"_s });
        return;
    }

    if (document->page() && !document->page()->isVisible()) {
        promise->reject(Exception { ExceptionCode::SecurityError, "Only visible documents can lock the screen orientation"_s });
        return;
    }

    if (document->settings().fullscreenRequirementForScreenOrientationLockingEnabled()) {
#if ENABLE(FULLSCREEN_API)
        if (CheckedPtr fullscreenManager = document->fullscreenManagerIfExists(); !fullscreenManager || !fullscreenManager->isFullscreen()) {
#else
        if (true) {
#endif
            promise->reject(Exception { ExceptionCode::SecurityError, "Locking the screen orientation is only allowed when in fullscreen"_s });
            return;
        }
    }
    if (!isSupportedLockType(lockType)) {
        promise->reject(Exception { ExceptionCode::NotSupportedError, "Lock type should be one of { \"any\", \"natural\", \"portrait\", \"landscape\" }"_s });
        return;
    }
    if (auto previousPromise = manager->takeLockPromise()) {
        queueTaskKeepingObjectAlive(*this, TaskSource::DOMManipulation, [previousPromise = WTFMove(previousPromise)]() mutable {
            previousPromise->reject(Exception { ExceptionCode::AbortError, "A new lock request was started"_s });
        });
    }
    manager->setLockPromise(*this, WTFMove(promise));
    manager->lock(lockType, [this, protectedThis = makePendingActivity(*this)](std::optional<Exception>&& exception) mutable {
        queueTaskKeepingObjectAlive(*this, TaskSource::DOMManipulation, [this, exception = WTFMove(exception)]() mutable {
            auto* manager = this->manager();
            if (!manager)
                return;

            auto promise = manager->takeLockPromise();
            if (!promise)
                return;

            if (exception)
                promise->reject(WTFMove(*exception));
            else
                promise->resolve();
        });
    });
}

ExceptionOr<void> ScreenOrientation::unlock()
{
    auto* document = this->document();
    if (!document || !document->isFullyActive())
        return Exception { ExceptionCode::InvalidStateError, "Document is not fully active."_s };

    if (!document->isSameOriginAsTopDocument())
        return { };

    if (document->page() && !document->page()->isVisible())
        return Exception { ExceptionCode::SecurityError, "Only visible documents can unlock the screen orientation"_s };

    if (auto* manager = this->manager())
        manager->unlock();
    return { };
}

auto ScreenOrientation::type() const -> Type
{
    auto* manager = this->manager();
    if (!manager)
        return naturalScreenOrientationType();
    return manager->currentOrientation();
}

uint16_t ScreenOrientation::angle() const
{
    auto* manager = this->manager();
    auto orientation = manager ? manager->currentOrientation() : naturalScreenOrientationType();

    // https://w3c.github.io/screen-orientation/#dfn-screen-orientation-values-table
    if (isPortrait(naturalScreenOrientationType())) {
        switch (orientation) {
        case Type::PortraitPrimary:
            return 0;
        case Type::PortraitSecondary:
            return 180;
        case Type::LandscapePrimary:
            return 90;
        case Type::LandscapeSecondary:
            return 270;
        }
    } else {
        switch (orientation) {
        case Type::PortraitPrimary:
            return 90;
        case Type::PortraitSecondary:
            return 270;
        case Type::LandscapePrimary:
            return 0;
        case Type::LandscapeSecondary:
            return 180;
        }
    }
    ASSERT_NOT_REACHED();
    return 0;
}

void ScreenOrientation::visibilityStateChanged()
{
    auto* document = this->document();
    if (!document)
        return;
    auto* manager = this->manager();
    if (!manager)
        return;

    if (shouldListenForChangeNotification())
        manager->addObserver(*this);
    else
        manager->removeObserver(*this);
}

bool ScreenOrientation::shouldListenForChangeNotification() const
{
    auto* document = this->document();
    if (!document || !document->frame())
        return false;
    return document->visibilityState() == VisibilityState::Visible;
}

void ScreenOrientation::screenOrientationDidChange(ScreenOrientationType)
{
    queueTaskToDispatchEvent(*this, TaskSource::DOMManipulation, Event::create(eventNames().changeEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

void ScreenOrientation::suspend(ReasonForSuspension)
{
    if (auto* manager = this->manager())
        manager->removeObserver(*this);
}

void ScreenOrientation::resume()
{
    if (!shouldListenForChangeNotification())
        return;
    if (auto* manager = this->manager())
        manager->addObserver(*this);
}

void ScreenOrientation::stop()
{
    auto* manager = this->manager();
    if (!manager)
        return;

    manager->removeObserver(*this);
    if (manager->lockRequester() == this) {
        queueTaskKeepingObjectAlive(*this, TaskSource::DOMManipulation, [promise = manager->takeLockPromise()] {
            promise->reject(Exception { ExceptionCode::AbortError, "Document is no longer fully active"_s });
        });
    }
}

bool ScreenOrientation::virtualHasPendingActivity() const
{
    return m_hasChangeEventListener;
}

void ScreenOrientation::eventListenersDidChange()
{
    m_hasChangeEventListener = hasEventListeners(eventNames().changeEvent);
}

} // namespace WebCore
