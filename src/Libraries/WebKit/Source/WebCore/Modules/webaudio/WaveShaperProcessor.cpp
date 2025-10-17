/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 16, 2023.
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

#if ENABLE(WEB_AUDIO)

#include "WaveShaperProcessor.h"

#include "WaveShaperDSPKernel.h"
#include <JavaScriptCore/Float32Array.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WaveShaperProcessor);

WaveShaperProcessor::WaveShaperProcessor(float sampleRate, size_t numberOfChannels)
    : AudioDSPKernelProcessor(sampleRate, numberOfChannels)
{
}

WaveShaperProcessor::~WaveShaperProcessor()
{
    if (isInitialized())
        uninitialize();
}

std::unique_ptr<AudioDSPKernel> WaveShaperProcessor::createKernel()
{
    return makeUnique<WaveShaperDSPKernel>(this);
}

void WaveShaperProcessor::setCurveForBindings(Float32Array* curve)
{
    ASSERT(isMainThread());
    // This synchronizes with process().
    Locker locker { m_processLock };

    m_curve = curve;
}

void WaveShaperProcessor::setOversampleForBindings(OverSampleType oversample)
{
    ASSERT(isMainThread());
    // This synchronizes with process().
    Locker locker { m_processLock };

    m_oversample = oversample;

    if (oversample == OverSampleNone)
        return;

    for (auto& audioDSPKernel : m_kernels)
        static_cast<WaveShaperDSPKernel&>(*audioDSPKernel).lazyInitializeOversampling();
}

void WaveShaperProcessor::process(const AudioBus* source, AudioBus* destination, size_t framesToProcess)
{
    if (!isInitialized()) {
        destination->zero();
        return;
    }

    bool channelCountMatches = source->numberOfChannels() == destination->numberOfChannels() && source->numberOfChannels() == m_kernels.size();
    ASSERT(channelCountMatches);
    if (!channelCountMatches)
        return;

    // The audio thread can't block on this lock, so we use tryLock() instead.
    if (!m_processLock.tryLock()) {
        // Too bad - tryLock() failed. We must be in the middle of a setCurve() call.
        destination->zero();
        return;
    }
    Locker locker { AdoptLock, m_processLock };

    // For each channel of our input, process using the corresponding WaveShaperDSPKernel into the output channel.
    for (size_t i = 0; i < m_kernels.size(); ++i)
        static_cast<WaveShaperDSPKernel&>(*m_kernels[i]).process(source->channel(i)->span().first(framesToProcess), destination->channel(i)->mutableSpan());
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
