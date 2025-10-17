/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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
#include "NavigateEvent.h"

#include "AbortController.h"
#include "CommonVM.h"
#include "Element.h"
#include "ExceptionCode.h"
#include "HTMLBodyElement.h"
#include "HistoryController.h"
#include "LocalFrameView.h"
#include "Navigation.h"
#include "NavigationNavigationType.h"
#include <wtf/IsoMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(NavigateEvent);

NavigateEvent::NavigateEvent(const AtomString& type, const NavigateEvent::Init& init, EventIsTrusted isTrusted, AbortController* abortController)
    : Event(EventInterfaceType::NavigateEvent, type, init, isTrusted)
    , m_navigationType(init.navigationType)
    , m_destination(init.destination)
    , m_signal(init.signal)
    , m_formData(init.formData)
    , m_downloadRequest(init.downloadRequest)
    , m_canIntercept(init.canIntercept)
    , m_userInitiated(init.userInitiated)
    , m_hashChange(init.hashChange)
    , m_hasUAVisualTransition(init.hasUAVisualTransition)
    , m_abortController(abortController)
{
    Locker<JSC::JSLock> locker(commonVM().apiLock());
    m_info.setWeakly(init.info);
}

Ref<NavigateEvent> NavigateEvent::create(const AtomString& type, const NavigateEvent::Init& init, AbortController* abortController)
{
    return adoptRef(*new NavigateEvent(type, init, EventIsTrusted::Yes, abortController));
}

Ref<NavigateEvent> NavigateEvent::create(const AtomString& type, const NavigateEvent::Init& init)
{
    // FIXME: AbortController is required but JS bindings need to create it with one.
    return adoptRef(*new NavigateEvent(type, init, EventIsTrusted::No, nullptr));
}

// https://html.spec.whatwg.org/multipage/nav-history-apis.html#navigateevent-perform-shared-checks
ExceptionOr<void> NavigateEvent::sharedChecks(Document& document)
{
    if (!document.isFullyActive())
        return Exception { ExceptionCode::InvalidStateError, "Document is not fully active"_s };

    if (!isTrusted())
        return Exception { ExceptionCode::SecurityError, "Event is not trusted"_s };

    if (defaultPrevented())
        return Exception { ExceptionCode::InvalidStateError, "Event was already canceled"_s };

    return { };
}

// https://html.spec.whatwg.org/multipage/nav-history-apis.html#dom-navigateevent-intercept
ExceptionOr<void> NavigateEvent::intercept(Document& document, NavigationInterceptOptions&& options)
{
    if (auto checkResult = sharedChecks(document); checkResult.hasException())
        return checkResult;

    if (!canIntercept())
        return Exception { ExceptionCode::SecurityError, "Event is not interceptable"_s };

    if (!isBeingDispatched())
        return Exception { ExceptionCode::InvalidStateError, "Event is not being dispatched"_s };

    ASSERT(!m_interceptionState || m_interceptionState == InterceptionState::Intercepted);

    if (options.handler)
        m_handlers.append(options.handler.releaseNonNull());

    if (options.focusReset) {
        // FIXME: Print warning to console if it was already set.
        m_focusReset = options.focusReset;
    }

    if (options.scroll) {
        // FIXME: Print warning to console if it was already set.
        m_scrollBehavior = options.scroll;
    }

    m_interceptionState = InterceptionState::Intercepted;

    return { };
}

// https://html.spec.whatwg.org/multipage/nav-history-apis.html#process-scroll-behavior
void NavigateEvent::processScrollBehavior(Document& document)
{
    ASSERT(m_interceptionState == InterceptionState::Committed);
    m_interceptionState = InterceptionState::Scrolled;

    if (m_navigationType == NavigationNavigationType::Traverse || m_navigationType == NavigationNavigationType::Reload)
        document.frame()->loader().checkedHistory()->restoreScrollPositionAndViewState();
    else if (!document.frame()->view()->scrollToFragment(document.url())) {
        if (!document.url().hasFragmentIdentifier())
            document.frame()->view()->scrollTo({ 0, 0 });
    }
}

// https://html.spec.whatwg.org/multipage/nav-history-apis.html#dom-navigateevent-scroll
ExceptionOr<void> NavigateEvent::scroll(Document& document)
{
    auto checkResult = sharedChecks(document);
    if (checkResult.hasException())
        return checkResult;

    if (m_interceptionState != InterceptionState::Committed)
        return Exception { ExceptionCode::InvalidStateError, "Interception has not been committed"_s };

    processScrollBehavior(document);

    return { };
}

// https://html.spec.whatwg.org/multipage/nav-history-apis.html#potentially-process-scroll-behavior
void NavigateEvent::potentiallyProcessScrollBehavior(Document& document)
{
    ASSERT(m_interceptionState == InterceptionState::Committed || m_interceptionState == InterceptionState::Scrolled);
    if (m_interceptionState == InterceptionState::Scrolled || m_scrollBehavior == NavigationScrollBehavior::Manual)
        return;

    processScrollBehavior(document);
}

// https://html.spec.whatwg.org/multipage/nav-history-apis.html#navigateevent-finish
void NavigateEvent::finish(Document& document, InterceptionHandlersDidFulfill didFulfill, FocusDidChange focusChanged)
{
    ASSERT(m_interceptionState != InterceptionState::Intercepted && m_interceptionState != InterceptionState::Finished);
    if (!m_interceptionState)
        return;

    ASSERT(m_interceptionState == InterceptionState::Committed || m_interceptionState == InterceptionState::Scrolled);
    if (focusChanged == FocusDidChange::No && m_focusReset != NavigationFocusReset::Manual) {
        RefPtr documentElement = document.documentElement();
        ASSERT(documentElement);

        RefPtr<Element> focusTarget = documentElement->findAutofocusDelegate();
        if (!focusTarget)
            focusTarget = document.body();
        if (!focusTarget)
            focusTarget = documentElement;

        document.setFocusedElement(focusTarget.get());
    }

    if (didFulfill == InterceptionHandlersDidFulfill::Yes)
        potentiallyProcessScrollBehavior(document);

    m_interceptionState = InterceptionState::Finished;
}

} // namespace WebCore
