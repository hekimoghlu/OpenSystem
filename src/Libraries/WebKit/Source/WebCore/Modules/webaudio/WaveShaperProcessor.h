/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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

#include "AudioDSPKernel.h"
#include "AudioDSPKernelProcessor.h"
#include "AudioNode.h"
#include <JavaScriptCore/Forward.h>
#include <memory>
#include <wtf/Lock.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// WaveShaperProcessor is an AudioDSPKernelProcessor which uses WaveShaperDSPKernel objects to implement non-linear distortion effects.

class WaveShaperProcessor final : public AudioDSPKernelProcessor {
    WTF_MAKE_TZONE_ALLOCATED(WaveShaperProcessor);
public:
    enum OverSampleType {
        OverSampleNone,
        OverSample2x,
        OverSample4x
    };

    WaveShaperProcessor(float sampleRate, size_t numberOfChannels);

    virtual ~WaveShaperProcessor();

    std::unique_ptr<AudioDSPKernel> createKernel() final;

    void process(const AudioBus* source, AudioBus* destination, size_t framesToProcess) final;

    void setCurveForBindings(Float32Array*);
    Float32Array* curveForBindings() WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_curve.get(); } // Doesn't grab the lock, only safe to call on the main thread.
    Float32Array* curve() const WTF_REQUIRES_LOCK(m_processLock) { return m_curve.get(); }

    void setOversampleForBindings(OverSampleType);
    OverSampleType oversampleForBindings() const WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_oversample; } // Doesn't grab the lock, only safe to call on the main thread.
    OverSampleType oversample() const WTF_REQUIRES_LOCK(m_processLock) { return m_oversample; }

    Lock& processLock() const WTF_RETURNS_LOCK(m_processLock) { return m_processLock; }

private:
    Type processorType() const final { return Type::WaveShaper; }

    // m_curve represents the non-linear shaping curve.
    RefPtr<Float32Array> m_curve WTF_GUARDED_BY_LOCK(m_processLock);

    OverSampleType m_oversample WTF_GUARDED_BY_LOCK(m_processLock) { OverSampleNone };

    // This synchronizes process() with setCurve().
    mutable Lock m_processLock;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::WaveShaperProcessor) \
    static bool isType(const WebCore::AudioProcessor& processor) { return processor.processorType() == WebCore::AudioProcessor::Type::WaveShaper; } \
SPECIALIZE_TYPE_TRAITS_END()
