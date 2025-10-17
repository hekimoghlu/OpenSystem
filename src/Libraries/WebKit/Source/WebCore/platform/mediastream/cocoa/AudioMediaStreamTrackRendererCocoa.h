/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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

#if ENABLE(MEDIA_STREAM)

#include "AudioMediaStreamTrackRenderer.h"
#include "AudioMediaStreamTrackRendererUnit.h"
#include "CAAudioStreamDescription.h"
#include "Logging.h"
#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudioTypes.h>
#include <optional>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class AudioSampleDataSource;
class AudioSampleBufferList;
class BaseAudioMediaStreamTrackRendererUnit;

class AudioMediaStreamTrackRendererCocoa final : public AudioMediaStreamTrackRenderer {
    WTF_MAKE_TZONE_ALLOCATED(AudioMediaStreamTrackRendererCocoa);
public:
    static Ref<AudioMediaStreamTrackRenderer> create(Init&& init) { return adoptRef(*new AudioMediaStreamTrackRendererCocoa(WTFMove(init))); }
    ~AudioMediaStreamTrackRendererCocoa();

private:
    explicit AudioMediaStreamTrackRendererCocoa(Init&&);

    // AudioMediaStreamTrackRenderer
    void pushSamples(const WTF::MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t) final;
    void start(CompletionHandler<void()>&&) final;
    void stop() final;
    void clear() final;
    void setVolume(float) final;
    void setAudioOutputDevice(const String&) final;

    void reset();
    void setRegisteredDataSource(RefPtr<AudioSampleDataSource>&&);

    BaseAudioMediaStreamTrackRendererUnit& rendererUnit();

    std::optional<CAAudioStreamDescription> m_outputDescription;
    RefPtr<AudioSampleDataSource> m_dataSource; // Used in background thread.
    RefPtr<AudioSampleDataSource> m_registeredDataSource; // Used in main thread.
    bool m_shouldRecreateDataSource { false };
    WebCore::AudioMediaStreamTrackRendererUnit::ResetObserver m_resetObserver;
    String m_deviceID;
};

}

#endif // ENABLE(MEDIA_STREAM)
