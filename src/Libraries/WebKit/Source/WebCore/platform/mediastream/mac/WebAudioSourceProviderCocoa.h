/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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

#if ENABLE(WEB_AUDIO)

#include "CAAudioStreamDescription.h"
#include "WebAudioSourceProvider.h"
#include <CoreAudio/CoreAudioTypes.h>
#include <wtf/Lock.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

typedef struct AudioBufferList AudioBufferList;
typedef struct OpaqueAudioConverter* AudioConverterRef;
typedef struct AudioStreamBasicDescription AudioStreamBasicDescription;
typedef const struct opaqueCMFormatDescription *CMFormatDescriptionRef;
typedef struct opaqueCMSampleBuffer *CMSampleBufferRef;

namespace WTF {
class LoggerHelper;
}

namespace WebCore {

class AudioSampleDataSource;
class CAAudioStreamDescription;
class PlatformAudioData;
class WebAudioBufferList;

class WEBCORE_EXPORT WebAudioSourceProviderCocoa
    : public WebAudioSourceProvider {
public:
    WebAudioSourceProviderCocoa();
    ~WebAudioSourceProviderCocoa();

protected:
    void receivedNewAudioSamples(const PlatformAudioData&, const AudioStreamDescription&, size_t);

        void setPollSamplesCount(size_t);

private:
    virtual void hasNewClient(AudioSourceProviderClient*) = 0;
#if !RELEASE_LOG_DISABLED
    virtual WTF::LoggerHelper& loggerHelper() = 0;
#endif

    // AudioSourceProvider
    void provideInput(AudioBus*, size_t) final;
    void setClient(WeakPtr<AudioSourceProviderClient>&&) final;

    void prepare(const AudioStreamBasicDescription&);

    Lock m_lock;
    WeakPtr<AudioSourceProviderClient> m_client;

    std::optional<CAAudioStreamDescription> m_inputDescription;
    std::optional<CAAudioStreamDescription> m_outputDescription;
    std::unique_ptr<WebAudioBufferList> m_audioBufferList;
    RefPtr<AudioSampleDataSource> m_dataSource;

    size_t m_pollSamplesCount { 3 };
    uint64_t m_writeCount { 0 };
    uint64_t m_readCount { 0 };
};

inline void WebAudioSourceProviderCocoa::setPollSamplesCount(size_t count)
{
    m_pollSamplesCount = count;
}

}

#endif // ENABLE(WEB_AUDIO)
