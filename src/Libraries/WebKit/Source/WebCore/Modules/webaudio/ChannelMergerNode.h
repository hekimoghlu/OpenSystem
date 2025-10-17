/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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
#include "ChannelMergerOptions.h"

namespace WebCore {

class AudioContext;
    
class ChannelMergerNode final : public AudioNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ChannelMergerNode);
public:
    static ExceptionOr<Ref<ChannelMergerNode>> create(BaseAudioContext&, const ChannelMergerOptions& = { });
    
    // AudioNode
    void process(size_t framesToProcess) override;
    
    ExceptionOr<void> setChannelCount(unsigned) final;
    ExceptionOr<void> setChannelCountMode(ChannelCountMode) final;
    
private:
    double tailTime() const override { return 0; }
    double latencyTime() const override { return 0; }
    bool requiresTailProcessing() const final { return false; }

    ChannelMergerNode(BaseAudioContext&, unsigned numberOfInputs);
};

} // namespace WebCore
