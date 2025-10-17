/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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
#include "WebMockContentFilterManager.h"

#if ENABLE(CONTENT_FILTERING)

#include "NetworkConnectionToWebProcessMessages.h"
#include "NetworkProcessConnection.h"
#include "WebProcess.h"
#include <WebCore/MockContentFilterManager.h>
#include <WebCore/MockContentFilterSettings.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {

WebMockContentFilterManager& WebMockContentFilterManager::singleton()
{
    static NeverDestroyed<WebMockContentFilterManager> manager;
    return manager.get();
}

void WebMockContentFilterManager::startObservingSettings()
{
    WebCore::MockContentFilterManager::singleton().setClient(this);
}

void WebMockContentFilterManager::mockContentFilterSettingsChanged(WebCore::MockContentFilterSettings& settings)
{
    if (RefPtr connection = WebProcess::singleton().existingNetworkProcessConnection())
        connection->connection().send(Messages::NetworkConnectionToWebProcess::InstallMockContentFilter(settings), 0);
}

};

#endif
