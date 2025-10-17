/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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

#include "MediaPlayer.h"
#include <span>
#include <wtf/ThreadSafeRefCounted.h>

#if ENABLE(WEB_CODECS)

namespace WebCore {

enum class AudioSampleFormat;
class MediaSample;

class PlatformRawAudioData : public ThreadSafeRefCounted<PlatformRawAudioData> {
public:
    virtual ~PlatformRawAudioData() = default;
    static RefPtr<PlatformRawAudioData> create(std::span<const uint8_t>, AudioSampleFormat, float sampleRate, int64_t timestamp, size_t numberOfFrames, size_t numberOfChannels);
    static Ref<PlatformRawAudioData> create(Ref<MediaSample>&&);

    virtual constexpr MediaPlatformType platformType() const = 0;

    virtual AudioSampleFormat format() const = 0;
    virtual size_t sampleRate() const = 0;
    virtual size_t numberOfChannels() const = 0;
    virtual size_t numberOfFrames() const = 0;
    virtual std::optional<uint64_t> duration() const = 0;
    virtual int64_t timestamp() const = 0;

    virtual size_t memoryCost() const = 0;

    void copyTo(std::span<uint8_t>, AudioSampleFormat, size_t planeIndex, std::optional<size_t> frameOffset, std::optional<size_t> frameCount, unsigned long copyElementCount);
};

} // namespace WebCore

#endif // ENABLE(WEB_CODECS)
