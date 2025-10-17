/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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

#include "MediaRecorderPrivateWriter.h"
#include "NowPlayingManager.h"
#include <wtf/Forward.h>

namespace WebCore {

class AudioDestination;
class AudioIOCallback;
class CDMFactory;
class MediaRecorderPrivateWriter;
class MediaRecorderPrivateWriterListener;
class NowPlayingManager;

class WEBCORE_EXPORT MediaStrategy {
public:
#if ENABLE(WEB_AUDIO)
    virtual Ref<AudioDestination> createAudioDestination(
        AudioIOCallback&, const String& inputDeviceId, unsigned numberOfInputChannels, unsigned numberOfOutputChannels, float sampleRate) = 0;
#endif
    virtual std::unique_ptr<NowPlayingManager> createNowPlayingManager() const;
    void resetMediaEngines();
    virtual bool hasThreadSafeMediaSourceSupport() const;
#if ENABLE(MEDIA_SOURCE)
    virtual void enableMockMediaSource();
    bool mockMediaSourceEnabled() const;
    static void addMockMediaSourceEngine();
#endif
#if PLATFORM(COCOA) && ENABLE(MEDIA_RECORDER)
    virtual std::unique_ptr<MediaRecorderPrivateWriter> createMediaRecorderPrivateWriter(MediaRecorderContainerType, MediaRecorderPrivateWriterListener&) const;
#endif
protected:
    MediaStrategy();
    virtual ~MediaStrategy();
    bool m_mockMediaSourceEnabled { false };
};

} // namespace WebCore
