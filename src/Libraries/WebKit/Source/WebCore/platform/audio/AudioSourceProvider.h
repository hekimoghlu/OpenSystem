/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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

#include <wtf/WeakPtr.h>

namespace WebCore {

class AudioBus;
class AudioSourceProviderClient;
    
// Abstract base-class for a pull-model client.
class AudioSourceProvider {
public:
    // provideInput() gets called repeatedly to render time-slices of a continuous audio stream.
    virtual void provideInput(AudioBus* bus, size_t framesToProcess) = 0;

    // If a client is set, we call it back when the audio format is available or changes.
    virtual void setClient(WeakPtr<AudioSourceProviderClient>&&) { };

    virtual bool isHandlingAVPlayer() const { return false; }

    virtual ~AudioSourceProvider() = default;
};

} // WebCore
