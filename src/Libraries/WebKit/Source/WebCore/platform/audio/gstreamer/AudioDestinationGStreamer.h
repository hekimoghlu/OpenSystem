/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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

#include "AudioBus.h"
#include "AudioDestination.h"
#include "GRefPtrGStreamer.h"
#include <wtf/Condition.h>
#include <wtf/Forward.h>

namespace WebCore {

class AudioDestinationGStreamer : public AudioDestination, public RefCounted<AudioDestinationGStreamer> {
public:
    AudioDestinationGStreamer(AudioIOCallback&, unsigned long numberOfOutputChannels, float sampleRate);
    virtual ~AudioDestinationGStreamer();

    void ref() const final { return RefCounted<AudioDestinationGStreamer>::ref(); }
    void deref() const final { return RefCounted<AudioDestinationGStreamer>::deref(); }

    WEBCORE_EXPORT void start(Function<void(Function<void()>&&)>&& dispatchToRenderThread, CompletionHandler<void(bool)>&&) final;
    WEBCORE_EXPORT void stop(CompletionHandler<void(bool)>&&) final;

    bool isPlaying() override { return m_isPlaying; }
    unsigned framesPerBuffer() const final;

    bool handleMessage(GstMessage*);
    void notifyIsPlaying(bool);

protected:
    virtual void startRendering(CompletionHandler<void(bool)>&&);
    virtual void stopRendering(CompletionHandler<void(bool)>&&);

private:
    void notifyStartupResult(bool);
    void notifyStopResult(bool);

    RefPtr<AudioBus> m_renderBus;

    bool m_isPlaying { false };
    bool m_audioSinkAvailable { false };
    GRefPtr<GstElement> m_pipeline;
    GRefPtr<GstElement> m_src;
    CompletionHandler<void(bool)> m_startupCompletionHandler;
    CompletionHandler<void(bool)> m_stopCompletionHandler;
};

} // namespace WebCore
