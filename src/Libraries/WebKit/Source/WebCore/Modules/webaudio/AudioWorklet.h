/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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
#include "MessagePort.h"
#include "Worklet.h"
#include <wtf/WeakPtr.h>

namespace WebCore {

class AudioWorkletNode;
class BaseAudioContext;
class AudioWorkletMessagingProxy;

class AudioWorklet final : public Worklet {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AudioWorklet);
public:
    static Ref<AudioWorklet> create(BaseAudioContext&);

    AudioWorkletMessagingProxy* proxy() const;
    BaseAudioContext* audioContext() const;

    void createProcessor(const String& name, TransferredMessagePort, Ref<SerializedScriptValue>&&, AudioWorkletNode&);

private:
    explicit AudioWorklet(BaseAudioContext&);

    // Worklet.
    Vector<Ref<WorkletGlobalScopeProxy>> createGlobalScopes() final;

    WeakPtr<BaseAudioContext, WeakPtrImplWithEventTargetData> m_audioContext;
};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
