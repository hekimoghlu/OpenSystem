/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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

#if ENABLE(VIDEO)

#include "ScriptExecutionContextIdentifier.h"
#include "TrackPrivateBaseClient.h"
#include <wtf/Lock.h>
#include <wtf/LoggerHelper.h>
#include <wtf/MediaTime.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/Vector.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

using TrackID = uint64_t;

class WEBCORE_EXPORT TrackPrivateBase
    : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<TrackPrivateBase>
#if !RELEASE_LOG_DISABLED
    , public LoggerHelper
#endif
{
    WTF_MAKE_TZONE_ALLOCATED(TrackPrivateBase);
    WTF_MAKE_NONCOPYABLE(TrackPrivateBase);
public:
    virtual ~TrackPrivateBase() = default;

    size_t addClient(TrackPrivateBaseClient::Dispatcher&&, TrackPrivateBaseClient&);
    void removeClient(uint32_t); // Can be called multiple times with the same id.

    virtual TrackID id() const { return 0; }
    virtual AtomString label() const { return emptyAtom(); }
    virtual AtomString language() const { return emptyAtom(); }

    virtual int trackIndex() const { return 0; }
    virtual std::optional<AtomString> trackUID() const;
    virtual std::optional<bool> defaultEnabled() const;

    virtual MediaTime startTimeVariance() const { return MediaTime::zeroTime(); }

    void willBeRemoved()
    {
        notifyClients([](auto& client) {
            client.willRemove();
        });
    }

    bool operator==(const TrackPrivateBase&) const;

    enum class Type { Video, Audio, Text };
    virtual Type type() const = 0;

#if !RELEASE_LOG_DISABLED
    virtual void setLogger(const Logger&, uint64_t);
    const Logger& logger() const final { ASSERT(m_logger); return *m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    WTFLogChannel& logChannel() const final;
#endif

    using Task = Function<void(TrackPrivateBaseClient&)>;
    void notifyClients(Task&&);
    void notifyMainThreadClient(Task&&);

protected:
    TrackPrivateBase() = default;

    template <typename T>
    class Shared final : public ThreadSafeRefCounted<Shared<T>> {
    public:
        static Ref<Shared> create(T&& obj) { return adoptRef(*new Shared(WTFMove(obj))); }

        T& get() { return m_obj; };
    private:
        Shared(T&& obj)
            : m_obj(WTFMove(obj))
        {
        }
        T m_obj;
    };
    using SharedDispatcher = Shared<TrackPrivateBaseClient::Dispatcher>;

    bool hasClients() const;
    bool hasOneClient() const;
    mutable Lock m_lock;
    using ClientRecord = std::tuple<RefPtr<SharedDispatcher>, WeakPtr<TrackPrivateBaseClient>, bool /* is main thread */>;
    Vector<ClientRecord> m_clients WTF_GUARDED_BY_LOCK(m_lock);

#if !RELEASE_LOG_DISABLED
    RefPtr<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
#endif
};

} // namespace WebCore

#endif
