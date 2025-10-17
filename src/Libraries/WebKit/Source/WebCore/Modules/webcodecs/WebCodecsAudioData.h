/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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

#if ENABLE(WEB_CODECS)

#include "AudioSampleFormat.h"
#include "BufferSource.h"
#include "ContextDestructionObserver.h"
#include "WebCodecsAudioInternalData.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class PlatformRawAudioData;
template<typename> class ExceptionOr;

class WebCodecsAudioData : public RefCounted<WebCodecsAudioData>, public ContextDestructionObserver {
public:
    ~WebCodecsAudioData();

    struct Init {
        AudioSampleFormat format { AudioSampleFormat::U8 };
        float sampleRate;
        int64_t timestamp { 0 };

        BufferSource data;
        size_t numberOfFrames;
        size_t numberOfChannels;
    };

    static ExceptionOr<Ref<WebCodecsAudioData>> create(ScriptExecutionContext&, Init&&);
    static Ref<WebCodecsAudioData> create(ScriptExecutionContext&, Ref<PlatformRawAudioData>&&);
    static Ref<WebCodecsAudioData> create(ScriptExecutionContext& context, WebCodecsAudioInternalData&& data) { return adoptRef(*new WebCodecsAudioData(context, WTFMove(data))); }

    std::optional<AudioSampleFormat> format() const;
    float sampleRate() const;
    size_t numberOfFrames() const;
    size_t numberOfChannels() const;
    std::optional<uint64_t> duration();
    int64_t timestamp() const;

    struct CopyToOptions {
        size_t planeIndex;
        std::optional<size_t> frameOffset { 0 };
        std::optional<size_t> frameCount;
        std::optional<AudioSampleFormat> format;
    };
    ExceptionOr<size_t> allocationSize(const CopyToOptions&);

    ExceptionOr<void> copyTo(BufferSource&&, CopyToOptions&&);
    ExceptionOr<Ref<WebCodecsAudioData>> clone(ScriptExecutionContext&);
    void close();

    bool isDetached() const { return m_isDetached; }

    const WebCodecsAudioInternalData& data() const { return m_data; }

    size_t memoryCost() const { return m_data.memoryCost(); }

private:
    explicit WebCodecsAudioData(ScriptExecutionContext&);
    WebCodecsAudioData(ScriptExecutionContext&, WebCodecsAudioInternalData&&);

    WebCodecsAudioInternalData m_data;
    bool m_isDetached { false };
};

} // namespace WebCore

#endif
