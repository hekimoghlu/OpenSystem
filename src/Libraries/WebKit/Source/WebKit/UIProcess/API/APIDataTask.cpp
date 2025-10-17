/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
#include "APIDataTask.h"

#include "APIDataTaskClient.h"
#include "NetworkProcessProxy.h"
#include "WebPageProxy.h"
#include <WebCore/ResourceError.h>

namespace API {

DataTask::~DataTask() = default;

Ref<DataTaskClient> DataTask::protectedClient() const
{
    return m_client;
}

void DataTask::setClient(Ref<DataTaskClient>&& client)
{
    m_client = WTFMove(client);
}

void DataTask::cancel()
{
    if (m_networkProcess && m_sessionID && m_identifier)
        m_networkProcess->cancelDataTask(*m_identifier, *m_sessionID);
    m_activity = nullptr;
}

void DataTask::networkProcessCrashed()
{
    m_activity = nullptr;
    m_client->didCompleteWithError(*this, WebCore::internalError(m_originalURL));
}

DataTask::DataTask(std::optional<WebKit::DataTaskIdentifier> identifier, WeakPtr<WebKit::WebPageProxy>&& page, WTF::URL&& originalURL, bool shouldRunAtForegroundPriority)
    : m_identifier(identifier)
    , m_page(WTFMove(page))
    , m_originalURL(WTFMove(originalURL))
    , m_networkProcess(m_page ? WeakPtr { m_page->websiteDataStore().networkProcess() } : nullptr)
    , m_sessionID(m_page ? std::optional<PAL::SessionID> { m_page->sessionID() } : std::nullopt)
    , m_client(DataTaskClient::create())
{
    if (RefPtr networkProcess = m_networkProcess.get())
        m_activity = shouldRunAtForegroundPriority ? networkProcess->throttler().foregroundActivity("WKDataTask"_s) : networkProcess->throttler().backgroundActivity("WKDataTask"_s);
}

void DataTask::didCompleteWithError(WebCore::ResourceError&& error)
{
    m_activity = nullptr;
    m_client->didCompleteWithError(*this, WTFMove(error));
}

} // namespace API
