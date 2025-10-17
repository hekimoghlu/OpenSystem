/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#include "WebNotification.h"

#include "APIDictionary.h"
#include <WebCore/NotificationData.h>

namespace WebKit {

WebNotification::WebNotification(const WebCore::NotificationData& data, std::optional<WebPageProxyIdentifier> pageIdentifier, const std::optional<WTF::UUID>& dataStoreIdentifier, IPC::Connection* sourceConnection)
    : m_data(data)
    , m_origin(API::SecurityOrigin::createFromString(data.originString))
    , m_pageIdentifier(pageIdentifier)
    , m_dataStoreIdentifier(dataStoreIdentifier)
    , m_sourceConnection(sourceConnection)
{
}

} // namespace WebKit
