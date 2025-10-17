/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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

#include "AudioDSPKernelProcessor.h"
#include "AudioParam.h"
#include <memory>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class AudioDSPKernel;
    
class DelayProcessor final : public AudioDSPKernelProcessor {
    WTF_MAKE_TZONE_ALLOCATED(DelayProcessor);
public:
    DelayProcessor(BaseAudioContext&, float sampleRate, unsigned numberOfChannels, double maxDelayTime);
    virtual ~DelayProcessor();
    
    std::unique_ptr<AudioDSPKernel> createKernel() override;
        
    AudioParam& delayTime() const { return m_delayTime.get(); }
    double maxDelayTime() { return delayTime().maxValue(); }

private:
    Type processorType() const final { return Type::Delay; }

    Ref<AudioParam> m_delayTime;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::DelayProcessor) \
    static bool isType(const WebCore::AudioProcessor& processor) { return processor.processorType() == WebCore::AudioProcessor::Type::Delay; } \
SPECIALIZE_TYPE_TRAITS_END()
