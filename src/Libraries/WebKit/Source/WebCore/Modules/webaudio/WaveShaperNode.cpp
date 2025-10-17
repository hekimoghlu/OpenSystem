/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
#include "WaveShaperNode.h"

#if ENABLE(WEB_AUDIO)

#include "AudioContext.h"
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/TypedArrayInlines.h>
#include <JavaScriptCore/TypedArrays.h>
#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WaveShaperNode);

ExceptionOr<Ref<WaveShaperNode>> WaveShaperNode::create(BaseAudioContext& context, const WaveShaperOptions& options)
{
    RefPtr<Float32Array> curve;
    if (options.curve) {
        curve = Float32Array::tryCreate(options.curve->data(), options.curve->size());
        if (!curve)
            return Exception { ExceptionCode::InvalidStateError, "Invalid curve parameter"_s };
    }

    auto node = adoptRef(*new WaveShaperNode(context));

    auto result = node->handleAudioNodeOptions(options, { 2, ChannelCountMode::Max, ChannelInterpretation::Speakers });
    if (result.hasException())
        return result.releaseException();

    if (curve) {
        result = node->setCurveForBindings(WTFMove(curve));
        if (result.hasException())
            return result.releaseException();
    }

    node->setOversampleForBindings(options.oversample);

    return node;
}

WaveShaperNode::WaveShaperNode(BaseAudioContext& context)
    : AudioBasicProcessorNode(context, NodeTypeWaveShaper)
{
    m_processor = makeUnique<WaveShaperProcessor>(context.sampleRate(), 1);

    initialize();
}

ExceptionOr<void> WaveShaperNode::setCurveForBindings(RefPtr<Float32Array>&& curve)
{
    ASSERT(isMainThread()); 
    DEBUG_LOG(LOGIDENTIFIER);
    if (curve && curve->length() < 2)
        return Exception { ExceptionCode::InvalidStateError, "Length of curve array cannot be less than 2"_s };

    if (curve) {
        // The specification states that we should maintain an internal copy of the curve so that
        // subsequent modifications of the contents of the array have no effect.
        auto clonedCurve = Float32Array::create(curve->data(), curve->length());
        curve = WTFMove(clonedCurve);
    }

    waveShaperProcessor()->setCurveForBindings(curve.get());
    return { };
}

RefPtr<Float32Array> WaveShaperNode::curveForBindings()
{
    ASSERT(isMainThread());
    RefPtr curve = waveShaperProcessor()->curveForBindings();
    if (!curve)
        return nullptr;

    // Make a clone of our internal array so that JS cannot modify our internal array
    // on the main thread while the audio thread is using it for rendering.
    return Float32Array::create(curve->data(), curve->length());
}

static inline WaveShaperProcessor::OverSampleType processorType(OverSampleType type)
{
    switch (type) {
    case OverSampleType::None:
        return WaveShaperProcessor::OverSampleNone;
    case OverSampleType::_2x:
        return WaveShaperProcessor::OverSample2x;
    case OverSampleType::_4x:
        return WaveShaperProcessor::OverSample4x;
    }
    ASSERT_NOT_REACHED();
    return WaveShaperProcessor::OverSampleNone;
}

void WaveShaperNode::setOversampleForBindings(OverSampleType type)
{
    ASSERT(isMainThread());
    INFO_LOG(LOGIDENTIFIER, type);

    // Synchronize with any graph changes or changes to channel configuration.
    Locker contextLocker { context().graphLock() };
    waveShaperProcessor()->setOversampleForBindings(processorType(type));
}

auto WaveShaperNode::oversampleForBindings() const -> OverSampleType
{
    ASSERT(isMainThread());
    switch (waveShaperProcessor()->oversampleForBindings()) {
    case WaveShaperProcessor::OverSampleNone:
        return OverSampleType::None;
    case WaveShaperProcessor::OverSample2x:
        return OverSampleType::_2x;
    case WaveShaperProcessor::OverSample4x:
        return OverSampleType::_4x;
    }
    ASSERT_NOT_REACHED();
    return OverSampleType::None;
}

bool WaveShaperNode::propagatesSilence() const
{
    if (!waveShaperProcessor()->processLock().tryLock())
        return false;

    Locker locker { AdoptLock, waveShaperProcessor()->processLock() };
    auto curve = waveShaperProcessor()->curve();
    return !curve || !curve->length();
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
