/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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

#include "EventLoop.h"
#include <wtf/Forward.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class XMLHttpRequestProgressEventThrottle;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::XMLHttpRequestProgressEventThrottle> : std::true_type { };
}

namespace WebCore {

class Event;
class XMLHttpRequest;

enum ProgressEventAction {
    DoNotFlushProgressEvent,
    FlushProgressEvent
};

// This implements the XHR2 progress event dispatching: "dispatch a progress event called progress
// about every 50ms or for every byte received, whichever is least frequent".
class XMLHttpRequestProgressEventThrottle : public CanMakeWeakPtr<XMLHttpRequestProgressEventThrottle> {
public:
    explicit XMLHttpRequestProgressEventThrottle(XMLHttpRequest&);
    virtual ~XMLHttpRequestProgressEventThrottle();

    void updateProgress(bool isAsync, bool lengthComputable, unsigned long long loaded, unsigned long long total);
    void dispatchReadyStateChangeEvent(Event&, ProgressEventAction = DoNotFlushProgressEvent);
    void dispatchProgressEvent(const AtomString&);
    void dispatchErrorProgressEvent(const AtomString&);

    void suspend();
    void resume();

private:
    static const Seconds minimumProgressEventDispatchingInterval;

    void dispatchThrottledProgressEventTimerFired();
    void flushProgressEvent();
    void dispatchEventWhenPossible(Event&);

    // Weak pointer to our XMLHttpRequest object as it is the one holding us.
    XMLHttpRequest& m_target;

    unsigned long long m_loaded { 0 };
    unsigned long long m_total { 0 };

    EventLoopTimerHandle m_dispatchThrottledProgressEventTimer;

    bool m_hasPendingThrottledProgressEvent { false };
    bool m_lengthComputable { false };
    bool m_shouldDeferEventsDueToSuspension { false };
};

} // namespace WebCore
