/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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

#include <stdlib.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class AudioBus;

// Pass in 0.0 for sampleRate to use the file's sample-rate, otherwise a sample-rate conversion to the requested
// sampleRate will be made (if it doesn't already match the file's sample-rate).
// The created buffer will have its sample-rate set correctly to the result.

RefPtr<AudioBus> createBusFromInMemoryAudioFile(std::span<const uint8_t> data, bool mixToMono, float sampleRate);

} // namespace WebCore
