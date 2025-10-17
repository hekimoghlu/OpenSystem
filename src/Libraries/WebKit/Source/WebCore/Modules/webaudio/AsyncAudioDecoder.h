/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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

#include "Exception.h"
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {
class ArrayBuffer;
}

namespace WebCore {
class AudioBuffer;

// AsyncAudioDecoder asynchronously decodes audio file data from an ArrayBuffer in a worker thread.
// Upon successful decoding, the DecodingTaskPromise will be resolved with the decoded AudioBuffer
// otherwise an Exception will be returned.
using DecodingTaskPromise = WTF::NativePromise<Ref<WebCore::AudioBuffer>, Exception>;

class AsyncAudioDecoder final {
    WTF_MAKE_TZONE_ALLOCATED(AsyncAudioDecoder);
    WTF_MAKE_NONCOPYABLE(AsyncAudioDecoder);
public:
    AsyncAudioDecoder();

    // Must be called on the main thread.
    Ref<DecodingTaskPromise> decodeAsync(Ref<JSC::ArrayBuffer>&& audioData, float sampleRate);

private:
    Ref<RunLoop> m_runLoop;
};

} // namespace WebCore
