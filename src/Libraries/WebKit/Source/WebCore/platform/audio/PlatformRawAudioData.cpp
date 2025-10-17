/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
#include "PlatformRawAudioData.h"

#include "AudioSampleFormat.h"
#include "NotImplemented.h"
#include <wtf/RefPtr.h>

#if ENABLE(WEB_CODECS)

namespace WebCore {

#if !USE(GSTREAMER) && !USE(AVFOUNDATION)
RefPtr<PlatformRawAudioData> PlatformRawAudioData::create(std::span<const uint8_t>, AudioSampleFormat, float, int64_t, size_t, size_t)
{
    notImplemented();
    return nullptr;
}

void PlatformRawAudioData::copyTo(std::span<uint8_t>, AudioSampleFormat, size_t, std::optional<size_t>, std::optional<size_t>, unsigned long)
{
    notImplemented();
}
#endif

} // namespace WebCore

#endif // ENABLE(WEB_CODECS)
