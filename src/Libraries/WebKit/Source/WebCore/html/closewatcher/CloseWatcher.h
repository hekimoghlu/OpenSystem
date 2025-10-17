/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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

#include "AbortSignal.h"
#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include <wtf/ListHashSet.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class KeyboardEvent;

class CloseWatcher final : public RefCounted<CloseWatcher>, public EventTarget, public ActiveDOMObject {
    WTF_MAKE_TZONE_ALLOCATED(CloseWatcher);
public:
    struct Options {
        RefPtr<AbortSignal> signal;
    };

    static ExceptionOr<Ref<CloseWatcher>> create(ScriptExecutionContext&, const Options&);

    explicit CloseWatcher(Document&);

    bool isActive() const { return m_active; }

    void requestClose();
    bool requestToClose();
    void close();
    void destroy();

    ScriptExecutionContext* scriptExecutionContext() const { return ActiveDOMObject::scriptExecutionContext(); }

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }
private:
    static Ref<CloseWatcher> establish(Document&);

    void stop() final { destroy(); }
    bool virtualHasPendingActivity() const final;

    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::CloseWatcher; }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    void eventListenersDidChange() final;

    bool canBeClosed() const;

    bool m_active { true };
    bool m_isRunningCancelAction { false };
    bool m_hasCancelEventListener { false };
    bool m_hasCloseEventListener { false };
    RefPtr<AbortSignal> m_signal;
    uint32_t m_signalAlgorithm { };
};

} // namespace WebCore
