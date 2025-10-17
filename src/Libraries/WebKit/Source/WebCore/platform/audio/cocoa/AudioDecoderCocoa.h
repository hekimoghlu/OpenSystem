/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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

#if ENABLE(WEB_CODECS) && USE(AVFOUNDATION)

#include "AudioDecoder.h"
#include "AudioStreamDescription.h"
#include "FourCC.h"
#include <wtf/Expected.h>
#include <wtf/TZoneMalloc.h>

namespace WTF {
class WorkQueue;
}

namespace WebCore {

class InternalAudioDecoderCocoa;

class AudioDecoderCocoa final : public AudioDecoder {
    WTF_MAKE_TZONE_ALLOCATED(AudioDecoderCocoa);

public:
    static Ref<CreatePromise> create(const String& codecName, const Config&, OutputCallback&&);

    ~AudioDecoderCocoa();

    static Expected<std::pair<FourCharCode, std::optional<AudioStreamDescription::PCMFormat>>, String> isCodecSupported(const StringView&);

    static WTF::WorkQueue& queueSingleton();

private:
    explicit AudioDecoderCocoa(OutputCallback&&);
    Ref<DecodePromise> decode(EncodedData&&) final;
    Ref<GenericPromise> flush() final;
    void reset() final;
    void close() final;

    const Ref<InternalAudioDecoderCocoa> m_internalDecoder;
};

}

#endif // ENABLE(WEB_CODECS) && USE(AVFOUNDATION)
