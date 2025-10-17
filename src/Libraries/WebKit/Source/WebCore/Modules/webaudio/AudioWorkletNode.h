/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 16, 2021.
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

#include "ActiveDOMObject.h"
#include "AudioNode.h"
#include <wtf/Lock.h>
#include <wtf/RobinHoodHashMap.h>

namespace JSC {
class JSGlobalObject;
} // namespace JSC

namespace WebCore {

class AudioParamMap;
class AudioWorkletProcessor;
class MessagePort;

struct AudioParamDescriptor;
struct AudioWorkletNodeOptions;

template<typename> class AudioArray;
typedef AudioArray<float> AudioFloatArray;

class AudioWorkletNode : public AudioNode, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AudioWorkletNode);
public:
    static ExceptionOr<Ref<AudioWorkletNode>> create(JSC::JSGlobalObject&, BaseAudioContext&, String&& name, AudioWorkletNodeOptions&&);
    ~AudioWorkletNode();

    // ActiveDOMObject.
    void ref() const final { AudioNode::ref(); }
    void deref() const final { AudioNode::deref(); }

    AudioParamMap& parameters() { return m_parameters.get(); }
    MessagePort& port() { return m_port.get(); }

    void setProcessor(RefPtr<AudioWorkletProcessor>&&);
    void initializeAudioParameters(const Vector<AudioParamDescriptor>&, const std::optional<Vector<KeyValuePair<String, double>>>& paramValues);

private:
    AudioWorkletNode(BaseAudioContext&, const String& name, AudioWorkletNodeOptions&&, Ref<MessagePort>&&);

    enum class ProcessorError { ConstructorError, ProcessError };
    void fireProcessorErrorOnMainThread(ProcessorError);
    void didFinishProcessingOnRenderingThread(bool threwException);

    // AudioNode.
    void process(size_t framesToProcess) final;
    double tailTime() const final { return m_tailTime; }
    double latencyTime() const final { return 0; }
    bool requiresTailProcessing() const final { return true; }
    void checkNumberOfChannelsForInput(AudioNodeInput*) final;
    void updatePullStatus() final;

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;

    String m_name;
    Ref<AudioParamMap> m_parameters;
    Ref<MessagePort> m_port;
    Lock m_processLock;
    RefPtr<AudioWorkletProcessor> m_processor; // Should only be used on the rendering thread.
    MemoryCompactLookupOnlyRobinHoodHashMap<String, std::unique_ptr<AudioFloatArray>> m_paramValuesMap;
    RefPtr<Thread> m_workletThread { nullptr };

    // Keeps the reference of AudioBus objects from AudioNodeInput and AudioNodeOutput in order
    // to pass them to AudioWorkletProcessor.
    Vector<RefPtr<AudioBus>> m_inputs;
    Vector<Ref<AudioBus>> m_outputs;

    double m_tailTime { std::numeric_limits<double>::infinity() };
    bool m_wasOutputChannelCountGiven { false };
};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
