/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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

#include "AudioBasicProcessorNode.h"
#include "IIRFilterOptions.h"
#include "IIRProcessor.h"
#include <JavaScriptCore/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class IIRFilterNode final : public AudioBasicProcessorNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(IIRFilterNode);
public:
    static ExceptionOr<Ref<IIRFilterNode>> create(ScriptExecutionContext&, BaseAudioContext&, IIRFilterOptions&&);

    ExceptionOr<void> getFrequencyResponse(Float32Array& frequencyHz, Float32Array& magResponse, Float32Array& phaseResponse);

private:
    IIRFilterNode(BaseAudioContext&, const Vector<double>& feedforward, const Vector<double>& feedback, bool isFilterStable);

    IIRProcessor* iirProcessor() { return downcast<IIRProcessor>(processor()); }
};

} // namespace WebCore
