/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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

#include "InternalObserver.h"
#include "VoidCallback.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class ScriptExecutionContext;
class JSSubscriptionObserverCallback;
class SubscriptionObserverCallback;
struct SubscriptionObserver;

class InternalObserverFromScript final : public InternalObserver {
public:
    static Ref<InternalObserverFromScript> create(ScriptExecutionContext&, RefPtr<JSSubscriptionObserverCallback>);
    static Ref<InternalObserverFromScript> create(ScriptExecutionContext&, SubscriptionObserver&);

    explicit InternalObserverFromScript(ScriptExecutionContext&, RefPtr<JSSubscriptionObserverCallback>);
    explicit InternalObserverFromScript(ScriptExecutionContext&, SubscriptionObserver&);

    void next(JSC::JSValue) final;
    void error(JSC::JSValue) final;
    void complete() final;

    void visitAdditionalChildren(JSC::AbstractSlotVisitor&) const final;

protected:
    // ActiveDOMObject
    void stop() final
    {
        m_next = nullptr;
        m_error = nullptr;
        m_complete = nullptr;
    }

private:
    RefPtr<SubscriptionObserverCallback> m_next;
    RefPtr<SubscriptionObserverCallback> m_error;
    RefPtr<VoidCallback> m_complete;
};

} // namespace WebCore
