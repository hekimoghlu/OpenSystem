/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 4, 2022.
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

#if ENABLE(MEDIA_STREAM)

#include "CoreAudioSharedUnit.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class CoreAudioSharedInternalUnit final :  public CoreAudioSharedUnit::InternalUnit {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(CoreAudioSharedInternalUnit);
public:
    static Expected<UniqueRef<InternalUnit>, OSStatus> create(bool shouldUseVPIO);
    CoreAudioSharedInternalUnit(CoreAudioSharedUnit::StoredAudioUnit&&, bool shouldUseVPIO);
    ~CoreAudioSharedInternalUnit() final;

private:
    OSStatus initialize() final;
    OSStatus uninitialize() final;
    OSStatus start() final;
    OSStatus stop() final;
    OSStatus set(AudioUnitPropertyID, AudioUnitScope, AudioUnitElement, const void*, UInt32) final;
    OSStatus get(AudioUnitPropertyID, AudioUnitScope, AudioUnitElement, void*, UInt32*) final;
    OSStatus render(AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32, UInt32, AudioBufferList*) final;
    OSStatus defaultInputDevice(uint32_t*) final;
    OSStatus defaultOutputDevice(uint32_t*) final;
    bool setVoiceActivityDetection(bool) final;
    bool canRenderAudio() const final { return m_shouldUseVPIO; }

    CoreAudioSharedUnit::StoredAudioUnit m_audioUnit;
    bool m_shouldUseVPIO { false };
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
