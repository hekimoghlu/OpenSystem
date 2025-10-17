/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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

#include "AudioBasicProcessorNode.h"
#include "OverSampleType.h"
#include "WaveShaperOptions.h"
#include "WaveShaperProcessor.h"
#include <wtf/Forward.h>

namespace WebCore {

class WaveShaperNode final : public AudioBasicProcessorNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WaveShaperNode);
public:
    static ExceptionOr<Ref<WaveShaperNode>> create(BaseAudioContext&, const WaveShaperOptions& = { });

    // setCurve() is called on the main thread.
    ExceptionOr<void> setCurveForBindings(RefPtr<Float32Array>&&);
    RefPtr<Float32Array> curveForBindings();

    void setOversampleForBindings(OverSampleType);
    OverSampleType oversampleForBindings() const;

    double latency() const { return latencyTime(); }

private:    
    explicit WaveShaperNode(BaseAudioContext&);

    bool propagatesSilence() const final;

    WaveShaperProcessor* waveShaperProcessor() { return downcast<WaveShaperProcessor>(processor()); }
    const WaveShaperProcessor* waveShaperProcessor() const { return downcast<WaveShaperProcessor>(processor()); }
};

String convertEnumerationToString(WebCore::OverSampleType); // in JSOverSampleType.cpp

} // namespace WebCore

namespace WTF {
    
template<> struct LogArgument<WebCore::OverSampleType> {
    static String toString(WebCore::OverSampleType type) { return convertEnumerationToString(type); }
};
    
} // namespace WTF

#endif // ENABLE(WEB_AUDIO)
