/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 18, 2022.
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
#include "config.h"
#include "AudioDecoder.h"

#if ENABLE(WEB_CODECS)

#if USE(AVFOUNDATION)
#include "AudioDecoderCocoa.h"
#endif

#if USE(GSTREAMER)
#include "AudioDecoderGStreamer.h"
#include "GStreamerRegistryScanner.h"
#endif

#include <wtf/UniqueRef.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

bool AudioDecoder::isCodecSupported(const StringView& codec)
{
    bool isMPEG4AAC = codec == "mp4a.40.2"_s || codec == "mp4a.40.02"_s || codec == "mp4a.40.5"_s
        || codec == "mp4a.40.05"_s || codec == "mp4a.40.29"_s || codec == "mp4a.40.42"_s;
    bool isCodecAllowed = isMPEG4AAC || codec == "mp3"_s || codec == "opus"_s
        || codec == "alaw"_s || codec == "ulaw"_s || codec == "flac"_s
        || codec == "vorbis"_s || codec.startsWith("pcm-"_s);

    if (!isCodecAllowed)
        return false;

    bool result = false;
#if USE(GSTREAMER)
    auto& scanner = GStreamerRegistryScanner::singleton();
    result = scanner.isCodecSupported(GStreamerRegistryScanner::Configuration::Decoding, codec.toString());
#elif USE(AVFOUNDATION)
    result = !!AudioDecoderCocoa::isCodecSupported(codec);
#endif

    return result;
}

Ref<AudioDecoder::CreatePromise> AudioDecoder::create(const String& codecName, const Config& config, OutputCallback&& outputCallback)
{
#if USE(GSTREAMER)
    CreatePromise::Producer producer;
    Ref promise = producer.promise();
    CreateCallback callback = [producer = WTFMove(producer)] (auto&& result) mutable {
        producer.settle(WTFMove(result));
    };
    GStreamerAudioDecoder::create(codecName, config, WTFMove(callback), WTFMove(outputCallback));
    return promise;
#elif USE(AVFOUNDATION)
    return AudioDecoderCocoa::create(codecName, config, WTFMove(outputCallback));
#else
    UNUSED_PARAM(codecName);
    UNUSED_PARAM(config);
    UNUSED_PARAM(outputCallback);

    return CreatePromise::createAndReject("Not supported"_s));
#endif
}

AudioDecoder::AudioDecoder() = default;
AudioDecoder::~AudioDecoder() = default;

}

#endif // ENABLE(WEB_CODECS)
