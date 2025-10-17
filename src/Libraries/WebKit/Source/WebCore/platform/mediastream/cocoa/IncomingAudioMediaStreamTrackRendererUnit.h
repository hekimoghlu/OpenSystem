/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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

#if ENABLE(MEDIA_STREAM) && USE(LIBWEBRTC)

#include "BaseAudioMediaStreamTrackRendererUnit.h"
#include "CAAudioStreamDescription.h"
#include "LibWebRTCAudioModule.h"
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/LoggerHelper.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WTF {
class WorkQueue;
}

namespace WebCore {

class AudioMediaStreamTrackRendererInternalUnit;
class AudioSampleDataSource;
class AudioSampleBufferList;
class CAAudioStreamDescription;
class WebAudioBufferList;

class IncomingAudioMediaStreamTrackRendererUnit : public BaseAudioMediaStreamTrackRendererUnit
#if !RELEASE_LOG_DISABLED
    , public LoggerHelper
#endif
{
    WTF_MAKE_TZONE_ALLOCATED(IncomingAudioMediaStreamTrackRendererUnit);
public:
    explicit IncomingAudioMediaStreamTrackRendererUnit(LibWebRTCAudioModule&);
    ~IncomingAudioMediaStreamTrackRendererUnit();

    void newAudioChunkPushed(uint64_t);

    void ref() { m_audioModule.get()->ref(); };
    void deref() { m_audioModule.get()->deref(); };

private:
    struct Mixer;
    void start(Mixer&);
    void stop(Mixer&);
    void postTask(Function<void()>&&);
    void renderAudioChunk(uint64_t currentAudioSampleCount);

    // BaseAudioMediaStreamTrackRendererUnit
    void addResetObserver(const String&, ResetObserver&) final;
    void addSource(const String&, Ref<AudioSampleDataSource>&&) final;
    void removeSource(const String&, AudioSampleDataSource&) final;

    std::pair<bool, Vector<Ref<AudioSampleDataSource>>> addSourceToMixer(const String&, Ref<AudioSampleDataSource>&&);
    std::pair<bool, Vector<Ref<AudioSampleDataSource>>> removeSourceFromMixer(const String&, AudioSampleDataSource&);

#if !RELEASE_LOG_DISABLED
    // LoggerHelper.
    const Logger& logger() const final;
    ASCIILiteral logClassName() const final { return "IncomingAudioMediaStreamTrackRendererUnit"_s; }
    WTFLogChannel& logChannel() const final;
    uint64_t logIdentifier() const final;
#endif

    const ThreadSafeWeakPtr<LibWebRTCAudioModule> m_audioModule;
    const Ref<WTF::WorkQueue> m_queue;

    struct Mixer {
        UncheckedKeyHashSet<Ref<AudioSampleDataSource>> sources;
        RefPtr<AudioSampleDataSource> registeredMixedSource;
        String deviceID;
    };

    struct RenderMixer {
        Vector<Ref<AudioSampleDataSource>> inputSources;
        RefPtr<AudioSampleDataSource> mixedSource;
        size_t writeCount { 0 };
    };

    HashMap<String, Mixer> m_mixers WTF_GUARDED_BY_CAPABILITY(mainThread);

    // Background thread variables.
    HashMap<String, RenderMixer> m_renderMixers WTF_GUARDED_BY_CAPABILITY(m_queue.get());

    std::optional<CAAudioStreamDescription> m_outputStreamDescription WTF_GUARDED_BY_CAPABILITY(m_queue.get());
    std::unique_ptr<WebAudioBufferList> m_audioBufferList WTF_GUARDED_BY_CAPABILITY(m_queue.get());
    size_t m_sampleCount { 0 };

#if !RELEASE_LOG_DISABLED
    RefPtr<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

}

#endif // ENABLE(MEDIA_STREAM) && USE(LIBWEBRTC)
