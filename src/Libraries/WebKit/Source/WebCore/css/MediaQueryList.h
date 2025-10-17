/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "MediaQuery.h"
#include "MediaQueryMatcher.h"

namespace WebCore {

namespace MQ {
class MediaQueryEvaluator;
}

// MediaQueryList interface is specified at https://drafts.csswg.org/cssom-view/#the-mediaquerylist-interface
// The objects of this class are returned by window.matchMedia. They may be used to
// retrieve the current value of the given media query and to add/remove listeners that
// will be called whenever the value of the query changes.

class MediaQueryList final : public RefCounted<MediaQueryList>, public EventTarget, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaQueryList);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<MediaQueryList> create(Document&, MediaQueryMatcher&, MQ::MediaQueryList&&, bool matches);
    ~MediaQueryList();

    String media() const;
    bool matches();

    void addListener(RefPtr<EventListener>&&);
    void removeListener(RefPtr<EventListener>&&);

    void evaluate(MQ::MediaQueryEvaluator&, MediaQueryMatcher::EventMode);

    void detachFromMatcher();

private:
    MediaQueryList(Document&, MediaQueryMatcher&, MQ::MediaQueryList&&, bool matches);

    void setMatches(bool);

    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::MediaQueryList; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ContextDestructionObserver::scriptExecutionContext(); }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    void eventListenersDidChange() final;

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;

    RefPtr<MediaQueryMatcher> m_matcher;
    const MQ::MediaQueryList m_mediaQueries;
    const OptionSet<MQ::MediaQueryDynamicDependency> m_dynamicDependencies;
    unsigned m_evaluationRound; // Indicates if the query has been evaluated after the last style selector change.
    unsigned m_changeRound; // Used to know if the query has changed in the last style selector change.
    bool m_matches;
    bool m_hasChangeEventListener { false };
    bool m_needsNotification { false };
};

}
