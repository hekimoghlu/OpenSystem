/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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
#include "AudioWorkletThread.h"
#include "MessagePort.h"
#include "WorkletGlobalScope.h"
#include <wtf/RobinHoodHashMap.h>
#include <wtf/ThreadSafeWeakHashSet.h>

namespace JSC {
class VM;
}

namespace WebCore {

class AudioWorkletProcessorConstructionData;
class AudioWorkletProcessor;
class AudioWorkletThread;
class JSAudioWorkletProcessorConstructor;

struct WorkletParameters;

class AudioWorkletGlobalScope final : public WorkletGlobalScope {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AudioWorkletGlobalScope);
public:
    static RefPtr<AudioWorkletGlobalScope> tryCreate(AudioWorkletThread&, const WorkletParameters&);
    ~AudioWorkletGlobalScope();

    ExceptionOr<void> registerProcessor(String&& name, Ref<JSAudioWorkletProcessorConstructor>&&);
    RefPtr<AudioWorkletProcessor> createProcessor(const String& name, TransferredMessagePort, Ref<SerializedScriptValue>&& options);
    void processorIsNoLongerNeeded(AudioWorkletProcessor&);
    void visitProcessors(JSC::AbstractSlotVisitor&);

    size_t currentFrame() const { return m_currentFrame; }

    float sampleRate() const { return m_sampleRate; }

    double currentTime() const { return m_sampleRate > 0.0 ? m_currentFrame / static_cast<double>(m_sampleRate) : 0.0; }

    AudioWorkletThread& thread() const;
    void prepareForDestruction() final;

    std::unique_ptr<AudioWorkletProcessorConstructionData> takePendingProcessorConstructionData();

    void handlePreRenderTasks();
    void handlePostRenderTasks(size_t currentFrame);

    FetchOptions::Destination destination() const final { return FetchOptions::Destination::Audioworklet; }

private:
    AudioWorkletGlobalScope(AudioWorkletThread&, Ref<JSC::VM>&&, const WorkletParameters&);

    bool isAudioWorkletGlobalScope() const final { return true; }

    size_t m_currentFrame { 0 };
    const float m_sampleRate;
    MemoryCompactRobinHoodHashMap<String, RefPtr<JSAudioWorkletProcessorConstructor>> m_processorConstructorMap;
    ThreadSafeWeakHashSet<AudioWorkletProcessor> m_processors;
    std::unique_ptr<AudioWorkletProcessorConstructionData> m_pendingProcessorConstructionData;
    std::optional<JSC::VM::DrainMicrotaskDelayScope> m_delayMicrotaskDrainingDuringRendering;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AudioWorkletGlobalScope)
static bool isType(const WebCore::ScriptExecutionContext& context)
{
    auto* workletGlobalScope = dynamicDowncast<WebCore::WorkletGlobalScope>(context);
    return workletGlobalScope && workletGlobalScope->isAudioWorkletGlobalScope();
}
static bool isType(const WebCore::WorkletGlobalScope& context) { return context.isAudioWorkletGlobalScope(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WEB_AUDIO)
