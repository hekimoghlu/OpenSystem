/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
#include "InspectorFrontendRouter.h"

#include "InspectorFrontendChannel.h"
#include <wtf/Assertions.h>

namespace Inspector {

Ref<FrontendRouter> FrontendRouter::create()
{
    return adoptRef(*new FrontendRouter);
}

void FrontendRouter::connectFrontend(FrontendChannel& connection)
{
    if (m_connections.contains(&connection)) {
        ASSERT_NOT_REACHED();
        return;
    }

    m_connections.append(&connection);
}

void FrontendRouter::disconnectFrontend(FrontendChannel& connection)
{
    if (!m_connections.contains(&connection)) {
        ASSERT_NOT_REACHED();
        return;
    }

    m_connections.removeFirst(&connection);
}

void FrontendRouter::disconnectAllFrontends()
{
    m_connections.clear();
}

bool FrontendRouter::hasLocalFrontend() const
{
    for (auto* connection : m_connections) {
        if (connection->connectionType() == FrontendChannel::ConnectionType::Local)
            return true;
    }

    return false;
}

bool FrontendRouter::hasRemoteFrontend() const
{
    for (auto* connection : m_connections) {
        if (connection->connectionType() == FrontendChannel::ConnectionType::Remote)
            return true;
    }

    return false;
}

void FrontendRouter::sendEvent(const String& message) const
{
    for (auto* connection : m_connections)
        connection->sendMessageToFrontend(message);
}

void FrontendRouter::sendResponse(const String& message) const
{
    // FIXME: send responses to the appropriate frontend.
    for (auto* connection : m_connections)
        connection->sendMessageToFrontend(message);
}

} // namespace Inspector
