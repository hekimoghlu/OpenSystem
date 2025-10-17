/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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
#include "ModelProcessModelPlayerManager.h"

#if ENABLE(MODEL_PROCESS)

#include "ModelProcessModelPlayer.h"
#include "ModelProcessModelPlayerManagerProxyMessages.h"
#include "WebPage.h"
#include "WebProcess.h"
#include <WebCore/ModelPlayer.h>
#include <WebCore/ModelPlayerClient.h>
#include <WebCore/ModelPlayerIdentifier.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ModelProcessModelPlayerManager);

Ref<ModelProcessModelPlayerManager> ModelProcessModelPlayerManager::create()
{
    return adoptRef(*new ModelProcessModelPlayerManager());
}

ModelProcessModelPlayerManager::ModelProcessModelPlayerManager() = default;

ModelProcessModelPlayerManager::~ModelProcessModelPlayerManager() = default;

ModelProcessConnection& ModelProcessModelPlayerManager::modelProcessConnection()
{
    auto modelProcessConnection = m_modelProcessConnection.get();
    if (!modelProcessConnection) {
        modelProcessConnection = &WebProcess::singleton().ensureModelProcessConnection();
        m_modelProcessConnection = modelProcessConnection;
        modelProcessConnection = &WebProcess::singleton().ensureModelProcessConnection();
        modelProcessConnection->addClient(*this);
    }

    return *modelProcessConnection;
}

Ref<ModelProcessModelPlayer> ModelProcessModelPlayerManager::createModelProcessModelPlayer(WebPage& page, WebCore::ModelPlayerClient& client)
{
    auto identifier = WebCore::ModelPlayerIdentifier::generate();
    modelProcessConnection().connection().send(Messages::ModelProcessModelPlayerManagerProxy::CreateModelPlayer(identifier), 0);

    auto player = ModelProcessModelPlayer::create(identifier, page, client);
    m_players.add(identifier, player);

    return player;
}

void ModelProcessModelPlayerManager::deleteModelProcessModelPlayer(WebCore::ModelPlayer& modelPlayer)
{
    WebCore::ModelPlayerIdentifier identifier = modelPlayer.identifier();
    m_players.take(identifier);
    modelProcessConnection().connection().send(Messages::ModelProcessModelPlayerManagerProxy::DeleteModelPlayer(identifier), 0);
}

void ModelProcessModelPlayerManager::didReceivePlayerMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    if (const auto& player = m_players.get(WebCore::ModelPlayerIdentifier(decoder.destinationID())))
        player->didReceiveMessage(connection, decoder);
}

}

#endif // ENABLE(MODEL_PROCESS)
