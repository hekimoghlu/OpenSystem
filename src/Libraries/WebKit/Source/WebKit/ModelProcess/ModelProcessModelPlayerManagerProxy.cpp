/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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
#include "ModelProcessModelPlayerManagerProxy.h"

#if ENABLE(MODEL_PROCESS)

#include "ModelProcessModelPlayerProxy.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(ModelProcessModelPlayerManagerProxy);

ModelProcessModelPlayerManagerProxy::ModelProcessModelPlayerManagerProxy(ModelConnectionToWebProcess& connection)
    : m_modelConnectionToWebProcess(connection)
{
}

ModelProcessModelPlayerManagerProxy::~ModelProcessModelPlayerManagerProxy()
{
    clear();
}

std::optional<SharedPreferencesForWebProcess> ModelProcessModelPlayerManagerProxy::sharedPreferencesForWebProcess() const
{
    if (!m_modelConnectionToWebProcess)
        return std::nullopt;

    return m_modelConnectionToWebProcess->sharedPreferencesForWebProcess();
}

void ModelProcessModelPlayerManagerProxy::clear()
{
    auto proxies = std::exchange(m_proxies, { });

    for (auto& proxy : proxies.values())
        proxy->invalidate();
}

void ModelProcessModelPlayerManagerProxy::createModelPlayer(WebCore::ModelPlayerIdentifier identifier)
{
    ASSERT(RunLoop::isMain());
    ASSERT(m_modelConnectionToWebProcess);
    ASSERT(!m_proxies.contains(identifier));

    auto proxy = ModelProcessModelPlayerProxy::create(*this, identifier, m_modelConnectionToWebProcess->protectedConnection());
    m_proxies.add(identifier, WTFMove(proxy));
}

void ModelProcessModelPlayerManagerProxy::deleteModelPlayer(WebCore::ModelPlayerIdentifier identifier)
{
    ASSERT(RunLoop::isMain());

    if (auto proxy = m_proxies.take(identifier))
        proxy->invalidate();

    if (m_modelConnectionToWebProcess)
        m_modelConnectionToWebProcess->modelProcess().tryExitIfUnusedAndUnderMemoryPressure();
}

void ModelProcessModelPlayerManagerProxy::didReceivePlayerMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    ASSERT(RunLoop::isMain());
    if (auto* player = m_proxies.get(WebCore::ModelPlayerIdentifier(decoder.destinationID())))
        player->didReceiveMessage(connection, decoder);
}

} // namespace WebKit

#endif // ENABLE(MODEL_PROCESS)
