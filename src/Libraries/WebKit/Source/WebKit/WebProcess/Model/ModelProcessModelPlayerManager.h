/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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

#include "ModelProcessConnection.h"
#include <WebCore/ModelPlayerIdentifier.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class ModelPlayer;
class ModelPlayerClient;
}

namespace WebKit {

class WebPage;
class ModelProcessModelPlayer;

class ModelProcessModelPlayerManager
    : public ModelProcessConnection::Client
    , public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<ModelProcessModelPlayerManager> {
    WTF_MAKE_TZONE_ALLOCATED(ModelProcessModelPlayerManager);
public:
    static Ref<ModelProcessModelPlayerManager> create();
    ~ModelProcessModelPlayerManager();

    ModelProcessConnection& modelProcessConnection();

    Ref<ModelProcessModelPlayer> createModelProcessModelPlayer(WebPage&, WebCore::ModelPlayerClient&);
    void deleteModelProcessModelPlayer(WebCore::ModelPlayer&);

    void didReceivePlayerMessage(IPC::Connection&, IPC::Decoder&);

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

private:
    ModelProcessModelPlayerManager();

    HashMap<WebCore::ModelPlayerIdentifier, WeakPtr<ModelProcessModelPlayer>> m_players;
    ThreadSafeWeakPtr<ModelProcessConnection> m_modelProcessConnection;
};

}

#endif // ENABLE(MODEL_PROCESS)
