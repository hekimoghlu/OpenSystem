/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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

#include "SpeechRecognitionCaptureSourceImpl.h"
#include "SpeechRecognitionConnectionClientIdentifier.h"
#include <wtf/TZoneMalloc.h>

namespace WTF {
class MediaTime;
}

namespace WebCore {

class AudioStreamDescription;
class PlatformAudioData;
class SpeechRecognitionCaptureSourceImpl;
class SpeechRecognitionUpdate;

class SpeechRecognitionCaptureSource {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(SpeechRecognitionCaptureSource, WEBCORE_EXPORT);
public:
    SpeechRecognitionCaptureSource() = default;
    ~SpeechRecognitionCaptureSource() = default;
    WEBCORE_EXPORT void mute();

#if ENABLE(MEDIA_STREAM)
    using DataCallback = Function<void(const WTF::MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t)>;
    using StateUpdateCallback = Function<void(const SpeechRecognitionUpdate&)>;
    SpeechRecognitionCaptureSource(SpeechRecognitionConnectionClientIdentifier, DataCallback&&, StateUpdateCallback&&, Ref<RealtimeMediaSource>&&);
    WEBCORE_EXPORT static std::optional<WebCore::CaptureDevice> findCaptureDevice();
    WEBCORE_EXPORT static CaptureSourceOrError createRealtimeMediaSource(const CaptureDevice&, PageIdentifier);
#endif

private:
#if ENABLE(MEDIA_STREAM)
    std::unique_ptr<SpeechRecognitionCaptureSourceImpl> m_impl;
#endif
};

} // namespace WebCore
