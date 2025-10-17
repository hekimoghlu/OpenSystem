/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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

#include "IdleRequestCallback.h"
#include <wtf/Deque.h>
#include <wtf/MonotonicTime.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class IdleCallbackController;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::IdleCallbackController> : std::true_type { };
}

namespace WebCore {

class Document;
class WeakPtrImplWithEventTargetData;

class IdleCallbackController : public CanMakeWeakPtr<IdleCallbackController> {
    WTF_MAKE_TZONE_ALLOCATED(IdleCallbackController);

public:
    IdleCallbackController(Document&);

    int queueIdleCallback(Ref<IdleRequestCallback>&&, Seconds timeout);
    void removeIdleCallback(int);

    void startIdlePeriod();
    bool isEmpty() const { return m_idleRequestCallbacks.isEmpty() && m_runnableIdleCallbacks.isEmpty(); }

private:
    void queueTaskToInvokeIdleCallbacks();
    bool invokeIdleCallbacks();
    void invokeIdleCallbackTimeout(unsigned identifier);

    unsigned m_idleCallbackIdentifier { 0 };

    struct IdleRequest {
        unsigned identifier { 0 };
        Ref<IdleRequestCallback> callback;
        std::optional<MonotonicTime> timeout;
    };

    Deque<IdleRequest> m_idleRequestCallbacks;
    Deque<IdleRequest> m_runnableIdleCallbacks;
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
};

} // namespace WebCore
