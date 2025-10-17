/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

#include "ContextDestructionObserver.h"
#include "WebCoreOpaqueRoot.h"
#include <wtf/LoggerHelper.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class SourceBuffer;
class TrackListBase;
class TrackPrivateBase;
class TrackPrivateBaseClient;
using TrackID = uint64_t;

class TrackBase
    : public RefCounted<TrackBase>
    , public ContextDestructionObserver
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
    WTF_MAKE_TZONE_ALLOCATED(TrackBase);
public:
    virtual ~TrackBase();

    virtual void didMoveToNewDocument(Document&);

    enum Type { BaseTrack, TextTrack, AudioTrack, VideoTrack };
    Type type() const { return m_type; }

    virtual AtomString id() const { return m_id; }
    TrackID trackId() const { return m_trackId; }
    AtomString label() const { return m_label; }
    AtomString validBCP47Language() const { return m_validBCP47Language; }
    AtomString language() const { return m_language; }

    virtual int uniqueId() const { return m_uniqueId; }

#if ENABLE(MEDIA_SOURCE)
    SourceBuffer* sourceBuffer() const { return m_sourceBuffer; }
    void setSourceBuffer(SourceBuffer* buffer) { m_sourceBuffer = buffer; }
#endif

    void setTrackList(TrackListBase&);
    void clearTrackList();
    TrackListBase* trackList() const;
    WebCoreOpaqueRoot opaqueRoot();

    virtual bool enabled() const = 0;

#if !RELEASE_LOG_DISABLED
    virtual void setLogger(const Logger&, uint64_t);
    const Logger& logger() const final { ASSERT(m_logger); return *m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    WTFLogChannel& logChannel() const final;
#endif

protected:
    TrackBase(ScriptExecutionContext*, Type, const std::optional<AtomString>& id, TrackID, const AtomString& label, const AtomString& language);

    virtual void setId(TrackID id)
    {
        m_id = AtomString::number(id);
        m_trackId = id;
    }
    virtual void setLabel(const AtomString& label) { m_label = label; }
    virtual void setLanguage(const AtomString&);

#if ENABLE(MEDIA_SOURCE)
    SourceBuffer* m_sourceBuffer { nullptr };
#endif

    void addClientToTrackPrivateBase(TrackPrivateBaseClient&, TrackPrivateBase&);
    void removeClientFromTrackPrivateBase(TrackPrivateBase&);

private:
    Type m_type;
    int m_uniqueId;
    AtomString m_id;
    TrackID m_trackId { 0 };
    AtomString m_label;
    AtomString m_language;
    AtomString m_validBCP47Language;
#if !RELEASE_LOG_DISABLED
    RefPtr<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
#endif
    WeakPtr<TrackListBase, WeakPtrImplWithEventTargetData> m_trackList;
    size_t m_clientRegistrationId;
};

class MediaTrackBase : public TrackBase {
    WTF_MAKE_TZONE_ALLOCATED(MediaTrackBase);
public:
    const AtomString& kind() const { return m_kind; }
    virtual void setKind(const AtomString&);

protected:
    MediaTrackBase(ScriptExecutionContext*, Type, const std::optional<AtomString>& id, TrackID, const AtomString& label, const AtomString& language);

    void setKindInternal(const AtomString&);

private:
    virtual bool isValidKind(const AtomString&) const = 0;

    AtomString m_kind;
};

inline WebCoreOpaqueRoot root(TrackBase* track)
{
    return track->opaqueRoot();
}

} // namespace WebCore

#endif
