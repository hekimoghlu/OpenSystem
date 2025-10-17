/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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
#include "MediaQueryList.h"

#include "AddEventListenerOptions.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "EventNames.h"
#include "HTMLFrameOwnerElement.h"
#include "MediaQueryEvaluator.h"
#include "MediaQueryListEvent.h"
#include "MediaQueryParser.h"
#include "Quirks.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MediaQueryList);

MediaQueryList::MediaQueryList(Document& document, MediaQueryMatcher& matcher, MQ::MediaQueryList&& mediaQueries, bool matches)
    : ActiveDOMObject(&document)
    , m_matcher(&matcher)
    , m_mediaQueries(WTFMove(mediaQueries))
    , m_dynamicDependencies(MQ::MediaQueryEvaluator { matcher.mediaType() }.collectDynamicDependencies(m_mediaQueries))
    , m_evaluationRound(matcher.evaluationRound())
    , m_changeRound(m_evaluationRound - 1) // Any value that is not the same as m_evaluationRound would do.
    , m_matches(matches)
{
    matcher.addMediaQueryList(*this);
}

Ref<MediaQueryList> MediaQueryList::create(Document& document, MediaQueryMatcher& matcher, MQ::MediaQueryList&& mediaQueries, bool matches)
{
    auto list = adoptRef(*new MediaQueryList(document, matcher, WTFMove(mediaQueries), matches));
    list->suspendIfNeeded();
    return list;
}

MediaQueryList::~MediaQueryList()
{
    if (m_matcher)
        m_matcher->removeMediaQueryList(*this);
}

void MediaQueryList::detachFromMatcher()
{
    m_matcher = nullptr;
}

String MediaQueryList::media() const
{
    StringBuilder builder;
    MQ::serialize(builder, m_mediaQueries);
    return builder.toString();
}

void MediaQueryList::addListener(RefPtr<EventListener>&& listener)
{
    if (!listener)
        return;

    addEventListener(eventNames().changeEvent, listener.releaseNonNull(), { });
}

void MediaQueryList::removeListener(RefPtr<EventListener>&& listener)
{
    if (!listener)
        return;

    removeEventListener(eventNames().changeEvent, *listener, { });
}

void MediaQueryList::evaluate(MQ::MediaQueryEvaluator& evaluator, MediaQueryMatcher::EventMode eventMode)
{
    RELEASE_ASSERT(m_matcher);
    if (m_evaluationRound != m_matcher->evaluationRound())
        setMatches(evaluator.evaluate(m_mediaQueries));

    m_needsNotification = m_changeRound == m_matcher->evaluationRound() || m_needsNotification;
    if (!m_needsNotification || eventMode == MediaQueryMatcher::EventMode::Schedule)
        return;
    ASSERT(eventMode == MediaQueryMatcher::EventMode::DispatchNow);

    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    if (document && document->quirks().shouldSilenceMediaQueryListChangeEvents())
        return;

    dispatchEvent(MediaQueryListEvent::create(eventNames().changeEvent, media(), matches()));
    m_needsNotification = false;
}

void MediaQueryList::setMatches(bool newValue)
{
    ASSERT(m_matcher);
    m_evaluationRound = m_matcher->evaluationRound();

    if (newValue == m_matches)
        return;

    m_matches = newValue;
    m_changeRound = m_evaluationRound;
}

bool MediaQueryList::matches()
{
    if (!m_matcher)
        return m_matches;

    if (m_dynamicDependencies.contains(MQ::MediaQueryDynamicDependency::Viewport))  {
        if (RefPtr document = dynamicDowncast<Document>(scriptExecutionContext())) {
            if (RefPtr ownerElement = document->ownerElement()) {
                ownerElement->document().updateLayout();
                m_matcher->evaluateAll(MediaQueryMatcher::EventMode::Schedule);
            }
        }
    }

    if (m_evaluationRound != m_matcher->evaluationRound())
        setMatches(m_matcher->evaluate(m_mediaQueries));

    return m_matches;
}

void MediaQueryList::eventListenersDidChange()
{
    m_hasChangeEventListener = hasEventListeners(eventNames().changeEvent);
}

bool MediaQueryList::virtualHasPendingActivity() const
{
    return m_hasChangeEventListener && m_matcher;
}

}
