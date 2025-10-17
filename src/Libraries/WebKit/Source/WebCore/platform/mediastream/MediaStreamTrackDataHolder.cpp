/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 20, 2024.
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
#include "MediaStreamTrackDataHolder.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(MEDIA_STREAM)

namespace WebCore {

class PreventSourceFromEndingObserverWrapper : public ThreadSafeRefCounted<PreventSourceFromEndingObserverWrapper, WTF::DestructionThread::Main> {
public:
    static Ref<PreventSourceFromEndingObserverWrapper> create(Ref<RealtimeMediaSource>&& source)
    {
        auto wrapper = adoptRef(*new PreventSourceFromEndingObserverWrapper);
        wrapper->initialize(WTFMove(source));
        return wrapper;
    }

private:
    PreventSourceFromEndingObserverWrapper() = default;

    void initialize(Ref<RealtimeMediaSource>&& source)
    {
        ensureOnMainThread([protectedThis = Ref { *this }, source = WTFMove(source)] () mutable {
            protectedThis->m_observer = makeUnique<PreventSourceFromEndingObserver>(WTFMove(source));
        });
    }

    class PreventSourceFromEndingObserver final : public RealtimeMediaSourceObserver {
        WTF_MAKE_TZONE_ALLOCATED_INLINE(PreventSourceFromEndingObserver);
    public:
        explicit PreventSourceFromEndingObserver(Ref<RealtimeMediaSource>&& source)
            : m_source(WTFMove(source))
        {
            m_source->addObserver(*this);
        }

        ~PreventSourceFromEndingObserver()
        {
            m_source->removeObserver(*this);
        }

    private:
        bool preventSourceFromEnding() final { return true; }

        Ref<RealtimeMediaSource> m_source;
    };

    std::unique_ptr<PreventSourceFromEndingObserver> m_observer;
};

MediaStreamTrackDataHolder::MediaStreamTrackDataHolder(String&& trackId, String&& label, RealtimeMediaSource::Type type, CaptureDevice::DeviceType deviceType, bool isEnabled, bool isEnded, MediaStreamTrackHintValue contentHint, bool isProducingData, bool isMuted, bool isInterrupted, RealtimeMediaSourceSettings settings, RealtimeMediaSourceCapabilities capabilities, Ref<RealtimeMediaSource>&& source)
    : trackId(WTFMove(trackId))
    , label(WTFMove(label))
    , type(type)
    , deviceType(deviceType)
    , isEnabled(isEnabled)
    , isEnded(isEnded)
    , contentHint(contentHint)
    , isProducingData(isProducingData)
    , isMuted(isMuted)
    , isInterrupted(isInterrupted)
    , settings(WTFMove(settings))
    , capabilities(WTFMove(capabilities))
    , source(source.get())
    , preventSourceFromEndingObserverWrapper(PreventSourceFromEndingObserverWrapper::create(WTFMove(source)))
{
}

MediaStreamTrackDataHolder::~MediaStreamTrackDataHolder()
{
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
