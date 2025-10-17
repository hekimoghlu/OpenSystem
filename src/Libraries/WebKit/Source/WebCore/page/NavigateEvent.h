/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 11, 2021.
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

#include "AbortController.h"
#include "AbortSignal.h"
#include "DOMFormData.h"
#include "Event.h"
#include "EventInit.h"
#include "JSValueInWrappedObject.h"
#include "LocalDOMWindowProperty.h"
#include "NavigationDestination.h"
#include "NavigationInterceptHandler.h"
#include "NavigationNavigationType.h"

namespace WebCore {

enum class InterceptionState : uint8_t {
    Intercepted,
    Committed,
    Scrolled,
    Finished,
};

enum class InterceptionHandlersDidFulfill : bool {
    No,
    Yes
};

enum class FocusDidChange : bool {
    No,
    Yes
};

class NavigateEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(NavigateEvent);
public:
    struct Init : EventInit {
        NavigationNavigationType navigationType { NavigationNavigationType::Push };
        RefPtr<NavigationDestination> destination;
        RefPtr<AbortSignal> signal;
        RefPtr<DOMFormData> formData;
        String downloadRequest;
        JSC::JSValue info;
        bool canIntercept { false };
        bool userInitiated { false };
        bool hashChange { false };
        bool hasUAVisualTransition { false };
    };

    enum class NavigationFocusReset : bool {
        AfterTransition,
        Manual,
    };

    enum class NavigationScrollBehavior : bool {
        AfterTransition,
        Manual,
    };

    struct NavigationInterceptOptions {
        RefPtr<NavigationInterceptHandler> handler;
        std::optional<NavigationFocusReset> focusReset;
        std::optional<NavigationScrollBehavior> scroll;
    };

    static Ref<NavigateEvent> create(const AtomString& type, const Init&);
    static Ref<NavigateEvent> create(const AtomString& type, const Init&, AbortController*);

    NavigationNavigationType navigationType() const { return m_navigationType; }
    bool canIntercept() const { return m_canIntercept; }
    bool userInitiated() const { return m_userInitiated; }
    bool hashChange() const { return m_hashChange; }
    bool hasUAVisualTransition() const { return m_hasUAVisualTransition; }
    NavigationDestination* destination() { return m_destination.get(); }
    AbortSignal* signal() { return m_signal.get(); }
    DOMFormData* formData() { return m_formData.get(); }
    String downloadRequest() { return m_downloadRequest; }
    JSC::JSValue info() { return m_info.getValue(); }
    JSValueInWrappedObject& infoWrapper() { return m_info; }

    ExceptionOr<void> intercept(Document&, NavigationInterceptOptions&&);
    ExceptionOr<void> scroll(Document&);

    bool wasIntercepted() const { return m_interceptionState.has_value(); }
    void setCanIntercept(bool canIntercept) { m_canIntercept = canIntercept; }
    void setInterceptionState(InterceptionState interceptionState) { m_interceptionState = interceptionState; }

    void finish(Document&, InterceptionHandlersDidFulfill, FocusDidChange);

    Vector<Ref<NavigationInterceptHandler>>& handlers() { return m_handlers; }

private:
    NavigateEvent(const AtomString& type, const Init&, EventIsTrusted, AbortController*);

    ExceptionOr<void> sharedChecks(Document&);
    void potentiallyProcessScrollBehavior(Document&);
    void processScrollBehavior(Document&);

    NavigationNavigationType m_navigationType;
    RefPtr<NavigationDestination> m_destination;
    RefPtr<AbortSignal> m_signal;
    RefPtr<DOMFormData> m_formData;
    String m_downloadRequest;
    Vector<Ref<NavigationInterceptHandler>> m_handlers;
    JSValueInWrappedObject m_info;
    bool m_canIntercept { false };
    bool m_userInitiated { false };
    bool m_hashChange { false };
    bool m_hasUAVisualTransition { false };
    std::optional<InterceptionState> m_interceptionState;
    std::optional<NavigationFocusReset> m_focusReset;
    std::optional<NavigationScrollBehavior> m_scrollBehavior;
    RefPtr<AbortController> m_abortController;
};

} // namespace WebCore
