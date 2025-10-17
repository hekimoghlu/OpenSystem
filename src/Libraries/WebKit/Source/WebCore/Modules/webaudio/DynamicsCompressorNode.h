/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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

#include "AudioNode.h"
#include "AudioParam.h"
#include "DynamicsCompressorOptions.h"
#include <memory>

namespace WebCore {

class DynamicsCompressor;

class DynamicsCompressorNode final : public AudioNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DynamicsCompressorNode);
public:
    static ExceptionOr<Ref<DynamicsCompressorNode>> create(BaseAudioContext&, const DynamicsCompressorOptions& = { });

    ~DynamicsCompressorNode();

    // AudioNode
    void process(size_t framesToProcess) final;
    void processOnlyAudioParams(size_t framesToProcess) final;
    void initialize() final;
    void uninitialize() final;

    // Static compression curve parameters.
    AudioParam& threshold() { return m_threshold.get(); }
    AudioParam& knee() { return m_knee.get(); }
    AudioParam& ratio() { return m_ratio.get(); }
    AudioParam& attack() { return m_attack.get(); }
    AudioParam& release() { return m_release.get(); }

    // Amount by which the compressor is currently compressing the signal in decibels.
    float reduction() const { return m_reduction; }

    ExceptionOr<void> setChannelCount(unsigned) final;
    ExceptionOr<void> setChannelCountMode(ChannelCountMode) final;

protected:
    explicit DynamicsCompressorNode(BaseAudioContext&, const DynamicsCompressorOptions& = { });
    virtual void setReduction(float reduction) { m_reduction = reduction; }

private:
    double tailTime() const final;
    double latencyTime() const final;
    bool requiresTailProcessing() const final;

    float noiseInjectionMultiplier() const final { return 0.01; }

    std::unique_ptr<DynamicsCompressor> m_dynamicsCompressor;
    Ref<AudioParam> m_threshold;
    Ref<AudioParam> m_knee;
    Ref<AudioParam> m_ratio;
    Ref<AudioParam> m_attack;
    Ref<AudioParam> m_release;
    float m_reduction { 0 };
};

} // namespace WebCore
