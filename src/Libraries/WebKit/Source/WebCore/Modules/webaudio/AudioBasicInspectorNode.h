/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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

namespace WebCore {

// AudioBasicInspectorNode is an AudioNode with one input and one output where the output might not necessarily connect to another node's input.
// If the output is not connected to any other node, then the AudioBasicInspectorNode's processIfNecessary() function will be called automatically by
// AudioContext before the end of each render quantum so that it can inspect the audio stream.
class AudioBasicInspectorNode : public AudioNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AudioBasicInspectorNode);
public:
    AudioBasicInspectorNode(BaseAudioContext&, NodeType);

protected:
    bool m_needAutomaticPull { false }; // When setting to true, AudioBasicInspectorNode will be pulled automatically by AudioContext before the end of each render quantum.

private:
    void pullInputs(size_t framesToProcess) override;
    void checkNumberOfChannelsForInput(AudioNodeInput*) override;

    void updatePullStatus() override;
};

} // namespace WebCore
