/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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
#include "CloseWatcher.h"

#include "CloseWatcherManager.h"
#include "DocumentInlines.h"
#include "Event.h"
#include "EventNames.h"
#include "KeyboardEvent.h"
#include "LocalDOMWindow.h"
#include "ScriptExecutionContext.h"
#include <wtf/IsoMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CloseWatcher);

ExceptionOr<Ref<CloseWatcher>> CloseWatcher::create(ScriptExecutionContext& context, const Options& options)
{
    RefPtr document = dynamicDowncast<Document>(context);
    if (!document->isFullyActive())
        return Exception { ExceptionCode::InvalidStateError, "Document is not fully active."_s };

    Ref watcher = CloseWatcher::establish(*document);

    if (RefPtr signal = options.signal) {
        if (signal->aborted()) {
            watcher->m_active = false;
            Ref manager = document->protectedWindow()->closeWatcherManager();
            manager->remove(watcher.get());
        } else {
            watcher->m_signal = signal;
            watcher->m_signalAlgorithm = signal->addAlgorithm([weakWatcher = WeakPtr { watcher.get() }](JSC::JSValue) mutable {
                if (weakWatcher)
                    weakWatcher->destroy();
            });
        }
    }

    return watcher;
}

Ref<CloseWatcher> CloseWatcher::establish(Document& document)
{
    ASSERT(document.isFullyActive());

    Ref watcher = adoptRef(*new CloseWatcher(document));
    watcher->suspendIfNeeded();

    Ref manager = document.protectedWindow()->closeWatcherManager();

    manager->add(watcher);
    return watcher;
}

CloseWatcher::CloseWatcher(Document& document)
    : ActiveDOMObject(document)
{ }

void CloseWatcher::requestClose()
{
    requestToClose();
}

bool CloseWatcher::requestToClose()
{
    if (!canBeClosed())
        return true;

    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    Ref manager = document->protectedWindow()->closeWatcherManager();
    bool canPreventClose = manager->canPreventClose() && document->protectedWindow()->hasHistoryActionActivation();
    Ref cancelEvent = Event::create(eventNames().cancelEvent, Event::CanBubble::No, canPreventClose ? Event::IsCancelable::Yes : Event::IsCancelable::No);
    m_isRunningCancelAction = true;
    dispatchEvent(cancelEvent);
    m_isRunningCancelAction = false;
    if (cancelEvent->defaultPrevented()) {
        document->protectedWindow()->consumeHistoryActionUserActivation();
        return false;
    }

    close();
    return true;
}

void CloseWatcher::close()
{
    if (!canBeClosed())
        return;

    destroy();

    Ref closeEvent = Event::create(eventNames().closeEvent, Event::CanBubble::No, Event::IsCancelable::No);

    dispatchEvent(closeEvent);
}

bool CloseWatcher::canBeClosed() const
{
    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    return isActive() && !m_isRunningCancelAction && document && document->isFullyActive();
}

void CloseWatcher::destroy()
{
    if (!isActive())
        return;

    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    if (document && document->protectedWindow()) {
        Ref manager = document->protectedWindow()->closeWatcherManager();
        manager->remove(*this);
    }

    m_active = false;

    if (RefPtr signal = m_signal)
        signal->removeAlgorithm(m_signalAlgorithm);
}

void CloseWatcher::eventListenersDidChange()
{
    m_hasCancelEventListener = hasEventListeners(eventNames().cancelEvent);
    m_hasCloseEventListener = hasEventListeners(eventNames().closeEvent);
}

bool CloseWatcher::virtualHasPendingActivity() const
{
    return m_hasCancelEventListener || m_hasCloseEventListener;
}

} // namespace WebCore
