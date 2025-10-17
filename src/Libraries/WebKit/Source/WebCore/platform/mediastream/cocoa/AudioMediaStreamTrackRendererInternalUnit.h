/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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

#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudioTypes.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {

class AudioMediaStreamTrackRendererUnit;
class CAAudioStreamDescription;

class AudioMediaStreamTrackRendererInternalUnit {
public:
    virtual ~AudioMediaStreamTrackRendererInternalUnit() = default;

    virtual void ref() const = 0;
    virtual void deref() const = 0;

    class Client : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<Client> {
    public:
        virtual ~Client() = default;

        virtual OSStatus render(size_t sampleCount, AudioBufferList&, uint64_t sampleTime, double hostTime, AudioUnitRenderActionFlags&) = 0;
        virtual void reset() = 0;
    };
    WEBCORE_EXPORT static Ref<AudioMediaStreamTrackRendererInternalUnit> create(const String&, Client&);

    using CreateFunction = Ref<AudioMediaStreamTrackRendererInternalUnit>(*)(const String&, AudioMediaStreamTrackRendererInternalUnit::Client&);
    WEBCORE_EXPORT static void setCreateFunction(CreateFunction);
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void close() { };
    virtual void retrieveFormatDescription(CompletionHandler<void(std::optional<CAAudioStreamDescription>)>&&) = 0;
    virtual void setLastDeviceUsed(const String&) { }
};

}

#endif // ENABLE(MEDIA_STREAM)
