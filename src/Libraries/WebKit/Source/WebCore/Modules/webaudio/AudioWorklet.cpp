/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
#include "AudioWorklet.h"

#include "AudioWorkletGlobalScope.h"
#include "AudioWorkletMessagingProxy.h"
#include "AudioWorkletNode.h"
#include "AudioWorkletProcessor.h"
#include "BaseAudioContext.h"
#include "WorkerRunLoop.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(AudioWorklet);

Ref<AudioWorklet> AudioWorklet::create(BaseAudioContext& audioContext)
{
    auto worklet = adoptRef(*new AudioWorklet(audioContext));
    worklet->suspendIfNeeded();
    return worklet;
}

AudioWorklet::AudioWorklet(BaseAudioContext& audioContext)
    : Worklet(*audioContext.document())
    , m_audioContext(audioContext)
{
}

Vector<Ref<WorkletGlobalScopeProxy>> AudioWorklet::createGlobalScopes()
{
    // WebAudio uses a single global scope.
    return { AudioWorkletMessagingProxy::create(*this) };
}

AudioWorkletMessagingProxy* AudioWorklet::proxy() const
{
    auto& proxies = this->proxies();
    if (proxies.isEmpty())
        return nullptr;
    return downcast<AudioWorkletMessagingProxy>(proxies.first().ptr());
}

BaseAudioContext* AudioWorklet::audioContext() const
{
    return m_audioContext.get();
}

void AudioWorklet::createProcessor(const String& name, TransferredMessagePort port, Ref<SerializedScriptValue>&& options, AudioWorkletNode& node)
{
    RefPtr proxy = this->proxy();
    ASSERT(proxy);
    if (!proxy)
        return;

    proxy->postTaskForModeToWorkletGlobalScope([name = name.isolatedCopy(), port, options = WTFMove(options), node = Ref { node }](ScriptExecutionContext& context) mutable {
        node->setProcessor(downcast<AudioWorkletGlobalScope>(context).createProcessor(name, port, WTFMove(options)));
        callOnMainThread([node = WTFMove(node)] { });
    }, WorkerRunLoop::defaultMode());
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
