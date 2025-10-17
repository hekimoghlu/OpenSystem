/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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

#if ENABLE(MODEL_PROCESS)

#include "Connection.h"
#include "MessageReceiver.h"
#include "ModelConnectionToWebProcess.h"
#include "SharedPreferencesForWebProcess.h"
#include <WebCore/ModelPlayerIdentifier.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class ModelProcessModelPlayerProxy;

class ModelProcessModelPlayerManagerProxy
    : public RefCounted<ModelProcessModelPlayerManagerProxy>
    , public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(ModelProcessModelPlayerManagerProxy);
public:
    static Ref<ModelProcessModelPlayerManagerProxy> create(ModelConnectionToWebProcess& modelConnectionToWebProcess)
    {
        return adoptRef(*new ModelProcessModelPlayerManagerProxy(modelConnectionToWebProcess));
    }

    ~ModelProcessModelPlayerManagerProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

    ModelConnectionToWebProcess* modelConnectionToWebProcess() { return m_modelConnectionToWebProcess.get(); }
    void clear();

    void didReceiveMessageFromWebProcess(IPC::Connection& connection, IPC::Decoder& decoder) { didReceiveMessage(connection, decoder); }
    void didReceivePlayerMessage(IPC::Connection&, IPC::Decoder&);

private:
    explicit ModelProcessModelPlayerManagerProxy(ModelConnectionToWebProcess&);

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Messages
    void createModelPlayer(WebCore::ModelPlayerIdentifier);
    void deleteModelPlayer(WebCore::ModelPlayerIdentifier);

    HashMap<WebCore::ModelPlayerIdentifier, Ref<ModelProcessModelPlayerProxy>> m_proxies;
    WeakPtr<ModelConnectionToWebProcess> m_modelConnectionToWebProcess;
};

} // namespace WebKit

#endif // ENABLE(MODEL_PROCESS)
