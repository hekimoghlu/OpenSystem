/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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
#include "FrameLoadState.h"

namespace WebKit {

FrameLoadState::~FrameLoadState()
{
}

void FrameLoadState::addObserver(FrameLoadStateObserver& observer)
{
    auto result = m_observers.add(observer);
    ASSERT_UNUSED(result, result.isNewEntry);
}

void FrameLoadState::removeObserver(FrameLoadStateObserver& observer)
{
    auto result = m_observers.remove(observer);
    ASSERT_UNUSED(result, result);
}

void FrameLoadState::didStartProvisionalLoad(const URL& url)
{
    ASSERT(m_provisionalURL.isEmpty());

    m_state = State::Provisional;
    m_provisionalURL = url;

    forEachObserver([&url](FrameLoadStateObserver& observer) {
        observer.didReceiveProvisionalURL(url);
        observer.didStartProvisionalLoad(url);
    });
}

void FrameLoadState::didSuspend()
{
    m_state = State::Finished;
    m_provisionalURL = { };

    forEachObserver([](FrameLoadStateObserver& observer) {
        observer.didCancelProvisionalLoad();
    });
}

void FrameLoadState::didExplicitOpen(const URL& url)
{
    ASSERT(!url.isNull());
    m_provisionalURL = { };
    setURL(url);
}

void FrameLoadState::didReceiveServerRedirectForProvisionalLoad(const URL& url)
{
    ASSERT(m_state == State::Provisional);

    m_provisionalURL = url;

    forEachObserver([&url](FrameLoadStateObserver& observer) {
        observer.didReceiveProvisionalURL(url);
    });
}

void FrameLoadState::didFailProvisionalLoad()
{
    ASSERT(m_state == State::Provisional);

    m_state = State::Finished;
    m_provisionalURL = { };
    m_unreachableURL = m_lastUnreachableURL;

    forEachObserver([&](FrameLoadStateObserver& observer) {
        observer.didCancelProvisionalLoad();
        observer.didFailProvisionalLoad(m_unreachableURL);
    });
}

void FrameLoadState::didCommitLoad()
{
    ASSERT(m_state == State::Provisional);

    m_state = State::Committed;
    ASSERT(!m_provisionalURL.isNull());
    m_url = m_provisionalURL.isNull() ? aboutBlankURL() : m_provisionalURL;
    m_provisionalURL = { };

    forEachObserver([&](FrameLoadStateObserver& observer) {
        observer.didCommitProvisionalLoad();
        observer.didCommitProvisionalLoad(m_isMainFrame);
    });
}

void FrameLoadState::didFinishLoad()
{
    ASSERT(m_state == State::Committed);
    ASSERT(m_provisionalURL.isEmpty());

    m_state = State::Finished;

    forEachObserver([&](FrameLoadStateObserver& observer) {
        observer.didFinishLoad(m_isMainFrame, m_url);
    });
}

void FrameLoadState::didFailLoad()
{
    ASSERT(m_state == State::Committed);
    ASSERT(m_provisionalURL.isEmpty());

    m_state = State::Finished;
    forEachObserver([&](FrameLoadStateObserver& observer) {
        observer.didFailLoad(m_url);
    });
}

void FrameLoadState::didSameDocumentNotification(const URL& url)
{
    ASSERT(!url.isNull());
    setURL(url.isNull() ? aboutBlankURL() : url);
}

void FrameLoadState::setURL(const URL& url)
{
    m_url = url;
    forEachObserver([&url](FrameLoadStateObserver& observer) {
        observer.didCancelProvisionalLoad();
        observer.didReceiveProvisionalURL(url);
        observer.didCommitProvisionalLoad();
    });
}

void FrameLoadState::setUnreachableURL(const URL& unreachableURL)
{
    m_lastUnreachableURL = m_unreachableURL;
    m_unreachableURL = unreachableURL;
}

void FrameLoadState::forEachObserver(const Function<void(FrameLoadStateObserver&)>& callback)
{
    m_observers.forEach([&callback](FrameLoadStateObserver& observer) {
        callback(Ref { observer });
    });
}

} // namespace WebKit
