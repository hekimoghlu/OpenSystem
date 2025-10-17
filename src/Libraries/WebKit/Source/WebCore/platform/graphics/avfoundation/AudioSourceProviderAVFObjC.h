/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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

#if ENABLE(WEB_AUDIO) && USE(MEDIATOOLBOX)

#include "AudioSourceProvider.h"
#include <wtf/MediaTime.h>
#include <wtf/RetainPtr.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/TypeCasts.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS AVAssetTrack;
OBJC_CLASS AVPlayerItem;
OBJC_CLASS AVMutableAudioMix;

typedef const struct opaqueMTAudioProcessingTap *MTAudioProcessingTapRef;
typedef struct AudioBufferList AudioBufferList;
typedef struct AudioStreamBasicDescription AudioStreamBasicDescription;
typedef struct OpaqueAudioConverter* AudioConverterRef;
typedef uint32_t MTAudioProcessingTapFlags;
typedef signed long CMItemCount;

namespace WebCore {

class AudioStreamDescription;
class CAAudioStreamDescription;
class CARingBuffer;
class PlatformAudioData;

class AudioSourceProviderAVFObjC : public ThreadSafeRefCounted<AudioSourceProviderAVFObjC>, public AudioSourceProvider {
public:
    using WeakValueType = AudioSourceProviderAVFObjC;
    static RefPtr<AudioSourceProviderAVFObjC> create(AVPlayerItem*);
    virtual ~AudioSourceProviderAVFObjC();

    void setPlayerItem(AVPlayerItem *);
    void setAudioTrack(AVAssetTrack *);

    using AudioCallback = Function<void(uint64_t startFrame, uint64_t numberOfFrames)>;
    WEBCORE_EXPORT void setAudioCallback(AudioCallback&&);
    using ConfigureAudioStorageCallback = Function<std::unique_ptr<CARingBuffer>(const CAAudioStreamDescription&, size_t frameCount)>;
    WEBCORE_EXPORT void setConfigureAudioStorageCallback(ConfigureAudioStorageCallback&&);

    void recreateAudioMixIfNeeded();

private:
    AudioSourceProviderAVFObjC(AVPlayerItem *);

    void destroyMixIfNeeded();
    void createMixIfNeeded();

    // AudioSourceProvider
    void provideInput(AudioBus*, size_t framesToProcess) override;
    void setClient(WeakPtr<AudioSourceProviderClient>&&) override;
    bool isHandlingAVPlayer() const final { return true; }

    static void initCallback(MTAudioProcessingTapRef, void*, void**);
    static void finalizeCallback(MTAudioProcessingTapRef);
    static void prepareCallback(MTAudioProcessingTapRef, CMItemCount, const AudioStreamBasicDescription*);
    static void unprepareCallback(MTAudioProcessingTapRef);
    static void processCallback(MTAudioProcessingTapRef, CMItemCount, MTAudioProcessingTapFlags, AudioBufferList*, CMItemCount*, MTAudioProcessingTapFlags*);

    void prepare(CMItemCount maxFrames, const AudioStreamBasicDescription *processingFormat);
    void unprepare();
    void process(MTAudioProcessingTapRef, CMItemCount numberFrames, MTAudioProcessingTapFlags flagsIn, AudioBufferList *bufferListInOut, CMItemCount *numberFramesOut, MTAudioProcessingTapFlags *flagsOut);

    RetainPtr<AVPlayerItem> m_avPlayerItem;
    RetainPtr<AVAssetTrack> m_avAssetTrack;
    RetainPtr<AVMutableAudioMix> m_avAudioMix;
    RetainPtr<MTAudioProcessingTapRef> m_tap;
    RetainPtr<AudioConverterRef> m_converter;
    std::unique_ptr<AudioBufferList> m_list;
    std::unique_ptr<AudioStreamBasicDescription> m_tapDescription;
    std::unique_ptr<AudioStreamBasicDescription> m_outputDescription;
    std::unique_ptr<CARingBuffer> m_ringBuffer;

    MediaTime m_startTimeAtLastProcess;
    MediaTime m_endTimeAtLastProcess;
    uint64_t m_writeAheadCount { 0 };
    uint64_t m_readCount { 0 };
    enum { NoSeek = std::numeric_limits<uint64_t>::max() };
    uint64_t m_seekTo { NoSeek };
    bool m_paused { true };
    WeakPtr<AudioSourceProviderClient> m_client;
    WeakPtrFactory<AudioSourceProviderAVFObjC> m_weakFactory;

    class TapStorage;
    RefPtr<TapStorage> m_tapStorage;
    AudioCallback m_audioCallback;
    ConfigureAudioStorageCallback m_configureAudioStorageCallback;
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AudioSourceProviderAVFObjC)
    static bool isType(const WebCore::AudioSourceProvider& provider) { return provider.isHandlingAVPlayer(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WEB_AUDIO) && USE(MEDIATOOLBOX)
