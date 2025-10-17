/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
#include "AudioWorkletMessagingProxy.h"

#include "AudioWorklet.h"
#include "AudioWorkletGlobalScope.h"
#include "AudioWorkletThread.h"
#include "BaseAudioContext.h"
#include "CacheStorageConnection.h"
#include "Document.h"
#include "LocalFrame.h"
#include "Page.h"
#include "Settings.h"
#include "WebRTCProvider.h"
#include "WorkletParameters.h"
#include "WorkletPendingTasks.h"

namespace WebCore {

static WorkletParameters generateWorkletParameters(AudioWorklet& worklet)
{
    RefPtr document = worklet.document();
    auto jsRuntimeFlags = document->settings().javaScriptRuntimeFlags();
    RELEASE_ASSERT(document->sessionID());

    return {
        document->url(),
        jsRuntimeFlags,
        worklet.audioContext() ? worklet.audioContext()->sampleRate() : 0.0f,
        worklet.identifier(),
        *document->sessionID(),
        document->settingsValues(),
        document->referrerPolicy(),
        worklet.audioContext() ? !worklet.audioContext()->isOfflineContext() : false,
        document->advancedPrivacyProtections(),
        document->noiseInjectionHashSalt()
    };
}

AudioWorkletMessagingProxy::AudioWorkletMessagingProxy(AudioWorklet& worklet)
    : m_worklet(worklet)
    , m_document(*worklet.document())
    , m_workletThread(AudioWorkletThread::create(*this, generateWorkletParameters(worklet)))
{
    ASSERT(isMainThread());

    m_workletThread->start();
}

AudioWorkletMessagingProxy::~AudioWorkletMessagingProxy()
{
    m_workletThread->stop();
    m_workletThread->clearProxies();
}

bool AudioWorkletMessagingProxy::postTaskForModeToWorkletGlobalScope(ScriptExecutionContext::Task&& task, const String& mode)
{
    m_workletThread->runLoop().postTaskForMode(WTFMove(task), mode);
    return true;
}

RefPtr<CacheStorageConnection> AudioWorkletMessagingProxy::createCacheStorageConnection()
{
    ASSERT_NOT_REACHED();
    return nullptr;
}

RefPtr<RTCDataChannelRemoteHandlerConnection> AudioWorkletMessagingProxy::createRTCDataChannelRemoteHandlerConnection()
{
    ASSERT(isMainThread());
    if (!m_document->page())
        return nullptr;
    return m_document->page()->webRTCProvider().createRTCDataChannelRemoteHandlerConnection();
}

ScriptExecutionContextIdentifier AudioWorkletMessagingProxy::loaderContextIdentifier() const
{
    return m_document->identifier();
}

void AudioWorkletMessagingProxy::postTaskToLoader(ScriptExecutionContext::Task&& task)
{
    m_document->postTask(WTFMove(task));
}

void AudioWorkletMessagingProxy::postTaskToAudioWorklet(Function<void(AudioWorklet&)>&& task)
{
    m_document->postTask([protectedThis = Ref { *this }, task = WTFMove(task)](ScriptExecutionContext&) {
        if (protectedThis->m_worklet)
            task(*protectedThis->m_worklet);
    });
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
