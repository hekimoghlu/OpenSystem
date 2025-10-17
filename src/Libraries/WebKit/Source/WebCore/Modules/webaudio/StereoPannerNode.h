/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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

#if ENABLE(WEB_AUDIO)

#include "AudioNode.h"
#include "AudioParam.h"
#include "StereoPanner.h"
#include "StereoPannerOptions.h"

namespace WebCore {

class AudioContext;

class StereoPannerNode final : public AudioNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(StereoPannerNode);
public:
    static ExceptionOr<Ref<StereoPannerNode>> create(BaseAudioContext&, const StereoPannerOptions& = { });
    
    ~StereoPannerNode();
    
    AudioParam& pan() { return m_pan.get(); }
    
private:
    StereoPannerNode(BaseAudioContext&, float pan);
    
    // AudioNode
    void process(size_t framesToProcess) final;
    void processOnlyAudioParams(size_t framesToProcess) final;
    bool requiresTailProcessing() const final { return false; }

    ExceptionOr<void> setChannelCount(unsigned) final;
    ExceptionOr<void> setChannelCountMode(ChannelCountMode) final;
    
    double tailTime() const final { return 0; }
    double latencyTime() const final { return 0; }
    
    Ref<AudioParam> m_pan;
    AudioFloatArray m_sampleAccurateValues;
};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
