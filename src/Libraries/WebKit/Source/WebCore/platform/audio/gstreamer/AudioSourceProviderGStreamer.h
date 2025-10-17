/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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

#if ENABLE(WEB_AUDIO) && ENABLE(VIDEO) && USE(GSTREAMER)

#include "AudioSourceProvider.h"
#include "AudioSourceProviderClient.h"
#include "GRefPtrGStreamer.h"
#include "MainThreadNotifier.h"
#include "WebAudioSourceProvider.h"
#include <gst/gst.h>
#include <wtf/Forward.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>

#if ENABLE(MEDIA_STREAM)
#include "GStreamerAudioStreamDescription.h"
#include "MediaStreamTrackPrivate.h"
#endif

typedef struct _GstAdapter GstAdapter;
typedef struct _GstAppSink GstAppSink;

namespace WebCore {

class AudioSourceProviderGStreamer final : public WebAudioSourceProvider {
public:
    static Ref<AudioSourceProviderGStreamer> create()
    {
        return adoptRef(*new AudioSourceProviderGStreamer());
    }

#if ENABLE(MEDIA_STREAM)
    static Ref<AudioSourceProviderGStreamer> create(MediaStreamTrackPrivate& source)
    {
        return adoptRef(*new AudioSourceProviderGStreamer(source));
    }
    AudioSourceProviderGStreamer(MediaStreamTrackPrivate&);
#endif

    void configureAudioBin(GstElement* audioBin, GstElement* audioSink);

    void provideInput(AudioBus*, size_t framesToProcess) override;
    void setClient(WeakPtr<AudioSourceProviderClient>&&) override;
    const AudioSourceProviderClient* client() const { return m_client.get(); }

    void handleNewDeinterleavePad(GstPad*);
    void deinterleavePadsConfigured();
    void handleRemovedDeinterleavePad(GstPad*);

    GstFlowReturn handleSample(GstAppSink*, bool isPreroll);
    void clearAdapters();

private:
    AudioSourceProviderGStreamer();
    ~AudioSourceProviderGStreamer();

#if ENABLE(MEDIA_STREAM)
    WeakPtr<MediaStreamTrackPrivate> m_captureSource;
    RefPtr<MediaStreamPrivate> m_streamPrivate;
    GRefPtr<GstElement> m_pipeline;
#endif
    enum MainThreadNotification {
        DeinterleavePadsConfigured = 1 << 0,
    };
    Ref<MainThreadNotifier<MainThreadNotification>> m_notifier;
    GRefPtr<GstElement> m_audioSinkBin;
    WeakPtr<AudioSourceProviderClient> m_client;
    int m_deinterleaveSourcePads { 0 };
    UncheckedKeyHashMap<int, GRefPtr<GstAdapter>> m_adapters WTF_GUARDED_BY_LOCK(m_adapterLock);
    unsigned long m_deinterleavePadAddedHandlerId { 0 };
    unsigned long m_deinterleaveNoMorePadsHandlerId { 0 };
    unsigned long m_deinterleavePadRemovedHandlerId { 0 };
    Lock m_adapterLock;
};

}
#endif // ENABLE(WEB_AUDIO) && ENABLE(VIDEO) && USE(GSTREAMER)
