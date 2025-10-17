/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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
#include "NetworkStateNotifier.h"

#include <wtf/NeverDestroyed.h>

#if USE(WEB_THREAD)
#include "WebCoreThread.h"
#endif

namespace WebCore {

static const Seconds updateStateSoonInterval { 2_s };

static bool shouldSuppressThreadSafetyCheck()
{
#if USE(WEB_THREAD)
    return WebThreadIsEnabled();
#else
    return false;
#endif
}

NetworkStateNotifier& NetworkStateNotifier::singleton()
{
    RELEASE_ASSERT(shouldSuppressThreadSafetyCheck() || isMainThread());
    static NeverDestroyed<NetworkStateNotifier> networkStateNotifier;
    return networkStateNotifier;
}

NetworkStateNotifier::NetworkStateNotifier()
    : m_updateStateTimer([] {
        singleton().updateState();
    })
{
}

bool NetworkStateNotifier::onLine()
{
    if (!m_isOnLine)
        updateState();
    return m_isOnLine.value_or(true);
}

void NetworkStateNotifier::addListener(Function<void(bool)>&& listener)
{
    ASSERT(listener);
    if (m_listeners.isEmpty())
        startObserving();
    m_listeners.append(WTFMove(listener));
}

void NetworkStateNotifier::updateState()
{
    auto wasOnLine = m_isOnLine;
    updateStateWithoutNotifying();
    if (m_isOnLine == wasOnLine)
        return;
    for (auto& listener : m_listeners)
        listener(m_isOnLine.value());
}

void NetworkStateNotifier::updateStateSoon()
{
    m_updateStateTimer.startOneShot(updateStateSoonInterval);
}

} // namespace WebCore
