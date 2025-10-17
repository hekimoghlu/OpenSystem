/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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
#include "ChannelSplitterOptions.h"

namespace WebCore {

class AudioContext;
    
class ChannelSplitterNode final : public AudioNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ChannelSplitterNode);
public:
    static ExceptionOr<Ref<ChannelSplitterNode>> create(BaseAudioContext&, const ChannelSplitterOptions& = { });

    // AudioNode
    void process(size_t framesToProcess) override;
    
    ExceptionOr<void> setChannelCount(unsigned) final;
    ExceptionOr<void> setChannelCountMode(ChannelCountMode) final;
    ExceptionOr<void> setChannelInterpretation(ChannelInterpretation) final;

private:
    double tailTime() const override { return 0; }
    double latencyTime() const override { return 0; }
    bool requiresTailProcessing() const final { return false; }

    ChannelSplitterNode(BaseAudioContext&, unsigned numberOfOutputs);
};

} // namespace WebCore
