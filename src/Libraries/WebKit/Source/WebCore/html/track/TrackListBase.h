/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "WebCoreOpaqueRoot.h"
#include <wtf/Observer.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class TrackListBase;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
// FIXME: TrackListBase inherits from RefCounted, what gives?
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::TrackListBase> : std::true_type { };
}

namespace WebCore {

class TrackBase;
using TrackID = uint64_t;

class TrackListBase : public RefCounted<TrackListBase>, public EventTarget, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TrackListBase);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    virtual ~TrackListBase();

    enum Type { BaseTrackList, TextTrackList, AudioTrackList, VideoTrackList };
    Type type() const { return m_type; }

    virtual unsigned length() const;
    virtual bool contains(TrackBase&) const;
    virtual bool contains(TrackID) const;
    virtual void remove(TrackBase&, bool scheduleEvent = true);
    virtual void remove(TrackID, bool scheduleEvent = true);
    virtual RefPtr<TrackBase> find(TrackID) const;

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const override = 0;
    ScriptExecutionContext* scriptExecutionContext() const final { return ContextDestructionObserver::scriptExecutionContext(); }

    void didMoveToNewDocument(Document&);

    WebCoreOpaqueRoot opaqueRoot();

    using OpaqueRootObserver = WTF::Observer<WebCoreOpaqueRoot()>;
    void setOpaqueRootObserver(const OpaqueRootObserver& observer) { m_opaqueRootObserver = observer; };

    // Needs to be public so tracks can call it
    void scheduleChangeEvent();
    bool isChangeEventScheduled() const { return m_isChangeEventScheduled; }

    bool isAnyTrackEnabled() const;

protected:
    TrackListBase(ScriptExecutionContext*, Type);

    void scheduleAddTrackEvent(Ref<TrackBase>&&);
    void scheduleRemoveTrackEvent(Ref<TrackBase>&&);

    Vector<RefPtr<TrackBase>> m_inbandTracks;

private:
    void scheduleTrackEvent(const AtomString& eventName, Ref<TrackBase>&&);

    // EventTarget
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    Type m_type;
    WeakPtr<OpaqueRootObserver> m_opaqueRootObserver;
    bool m_isChangeEventScheduled { false };
};

inline WebCoreOpaqueRoot root(TrackListBase* trackList)
{
    return trackList->opaqueRoot();
}

} // namespace WebCore

#endif // ENABLE(VIDEO)
