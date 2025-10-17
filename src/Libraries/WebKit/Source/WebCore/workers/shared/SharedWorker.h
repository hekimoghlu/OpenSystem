/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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

#include "AbstractWorker.h"
#include "ActiveDOMObject.h"
#include "SharedWorkerKey.h"
#include "SharedWorkerObjectIdentifier.h"
#include "URLKeepingBlobAlive.h"
#include <wtf/Identified.h>
#include <wtf/MonotonicTime.h>

namespace WebCore {

class MessagePort;
class ResourceError;
class TrustedScriptURL;

struct WorkerOptions;

class SharedWorker final : public AbstractWorker, public ActiveDOMObject, public Identified<SharedWorkerObjectIdentifier> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SharedWorker);
public:
    static ExceptionOr<Ref<SharedWorker>> create(Document&, std::variant<RefPtr<TrustedScriptURL>, String>&&, std::optional<std::variant<String, WorkerOptions>>&&);
    ~SharedWorker();

    void ref() const final { AbstractWorker::ref(); }
    void deref() const final { AbstractWorker::deref(); }

    static SharedWorker* fromIdentifier(SharedWorkerObjectIdentifier);
    MessagePort& port() const { return m_port.get(); }

    const String& identifierForInspector() const { return m_identifierForInspector; }

    void didFinishLoading(const ResourceError&);

    // EventTarget.
    ScriptExecutionContext* scriptExecutionContext() const final;

private:
    SharedWorker(Document&, const SharedWorkerKey&, Ref<MessagePort>&&);

    // EventTarget.
    enum EventTargetInterfaceType eventTargetInterface() const final;

    // ActiveDOMObject.
    void stop() final;
    bool virtualHasPendingActivity() const final;
    void suspend(ReasonForSuspension) final;
    void resume() final;

    SharedWorkerKey m_key;
    Ref<MessagePort> m_port;
    String m_identifierForInspector;
    URLKeepingBlobAlive m_blobURLExtension;
    bool m_isActive { true };
    bool m_isSuspendedForBackForwardCache { false };
};

} // namespace WebCore
