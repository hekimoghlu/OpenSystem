/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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

#if ENABLE(WEB_AUDIO) && ENABLE(MEDIA_STREAM)

#include "MediaStreamTrackPrivate.h"
#include "RealtimeMediaSource.h"
#include "WebAudioSourceProviderCocoa.h"
#include <wtf/CheckedRef.h>
#include <wtf/MediaTime.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class MediaStreamTrackAudioSourceProviderCocoa final
    : public WebAudioSourceProviderCocoa
    , MediaStreamTrackPrivateObserver
    , RealtimeMediaSource::AudioSampleObserver
    , public CanMakeCheckedPtr<MediaStreamTrackAudioSourceProviderCocoa> {
    WTF_MAKE_TZONE_ALLOCATED(MediaStreamTrackAudioSourceProviderCocoa);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MediaStreamTrackAudioSourceProviderCocoa);
public:
    static Ref<MediaStreamTrackAudioSourceProviderCocoa> create(MediaStreamTrackPrivate&);
    ~MediaStreamTrackAudioSourceProviderCocoa();

private:
    explicit MediaStreamTrackAudioSourceProviderCocoa(MediaStreamTrackPrivate&);

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    // WebAudioSourceProviderCocoa
    void hasNewClient(AudioSourceProviderClient*) final;
#if !RELEASE_LOG_DISABLED
    WTF::LoggerHelper& loggerHelper() final { return m_source.get(); }
#endif

    // MediaStreamTrackPrivateObserver
    void trackEnded(MediaStreamTrackPrivate&) final { }
    void trackMutedChanged(MediaStreamTrackPrivate&) final { }
    void trackSettingsChanged(MediaStreamTrackPrivate&) final { }
    void trackEnabledChanged(MediaStreamTrackPrivate&) final;

    // RealtimeMediaSource::AudioSampleObserver
    void audioSamplesAvailable(const WTF::MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t) final;

    WeakPtr<MediaStreamTrackPrivate> m_captureSource;
    Ref<RealtimeMediaSource> m_source;
    bool m_enabled { true };
    bool m_connected { false };
};

}

#endif
