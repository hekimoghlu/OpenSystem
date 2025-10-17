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
#include "TrackPrivateBase.h"
#include <wtf/SharedTask.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(VIDEO)

#include "Logging.h"

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TrackPrivateBase);

std::optional<AtomString> TrackPrivateBase::trackUID() const
{
    return std::nullopt;
}

std::optional<bool> TrackPrivateBase::defaultEnabled() const
{
    return std::nullopt;
}

bool TrackPrivateBase::operator==(const TrackPrivateBase& track) const
{
    return id() == track.id()
        && label() == track.label()
        && language() == track.language()
        && trackIndex() == track.trackIndex()
        && trackUID() == track.trackUID()
        && defaultEnabled() == track.defaultEnabled()
        && startTimeVariance() == track.startTimeVariance();
}

void TrackPrivateBase::notifyClients(Task&& task)
{
    Ref sharedTask = createSharedTask<void(TrackPrivateBaseClient&)>(WTFMove(task));
    // We ensure not to hold the lock for too long by making a copy (which are cheap)
    // as we could potentially get a re-entrant call which would cause a deadlock.
    Vector<ClientRecord> clients;
    {
        Locker locker { m_lock };
        clients = m_clients;
    }
    for (auto& tuple : clients) {
        auto& [dispatcher, weakClient, mainThread] = tuple;
        if (dispatcher) {
            dispatcher->get()([weakClient = WTFMove(weakClient), sharedTask] {
                if (weakClient)
                    sharedTask->run(*weakClient);
            });
        }
    }
}

void TrackPrivateBase::notifyMainThreadClient(Task&& task)
{
    // There will only ever be one main thread client.
    // We call the first one found.
    Vector<ClientRecord> clients;
    {
        Locker locker { m_lock };
        clients = m_clients;
    }
    for (auto& tuple : clients) {
        auto& [dispatcher, weakClient, mainThread] = tuple;
        if (dispatcher && mainThread) {
            dispatcher->get()([weakClient = WTFMove(weakClient), task = WTFMove(task)] {
                if (weakClient)
                    task(*weakClient);
            });
            break;
        }
    }
}

size_t TrackPrivateBase::addClient(TrackPrivateBaseClient::Dispatcher&& dispatcher, TrackPrivateBaseClient& client)
{
    Locker locker { m_lock };
    size_t index = m_clients.size();
    m_clients.append(std::make_tuple(SharedDispatcher::create(WTFMove(dispatcher)), WeakPtr { client }, isMainThread()));
    return index;
}

void TrackPrivateBase::removeClient(uint32_t index)
{
    Locker locker { m_lock };
    if (m_clients.size() > index)
        return;
    m_clients[index] = std::make_tuple<RefPtr<SharedDispatcher>, WeakPtr<TrackPrivateBaseClient>, bool>({ }, { }, false);
}

bool TrackPrivateBase::hasClients() const
{
    Locker locker { m_lock };
    return m_clients.size();
}

bool TrackPrivateBase::hasOneClient() const
{
    Locker locker { m_lock };
    return m_clients.size() == 1;
}

#if !RELEASE_LOG_DISABLED

static uint64_t s_uniqueId = 0;

void TrackPrivateBase::setLogger(const Logger& logger, uint64_t logIdentifier)
{
    m_logger = &logger;
    m_logIdentifier = childLogIdentifier(logIdentifier, ++s_uniqueId);
}

WTFLogChannel& TrackPrivateBase::logChannel() const
{
    return LogMedia;
}
#endif

} // namespace WebCore

#endif
