/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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

#include "ThreadableLoaderClient.h"
#include <wtf/Ref.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class ThreadableLoaderClientWrapper : public ThreadSafeRefCounted<ThreadableLoaderClientWrapper> {
public:
    static Ref<ThreadableLoaderClientWrapper> create(ThreadableLoaderClient& client, const String& initiator)
    {
        return adoptRef(*new ThreadableLoaderClientWrapper(client, initiator));
    }

    void clearClient()
    {
        m_done = true;
        m_client = nullptr;
    }

    bool done() const
    {
        return m_done;
    }

    void didSendData(unsigned long long bytesSent, unsigned long long totalBytesToBeSent)
    {
        if (m_client)
            m_client->didSendData(bytesSent, totalBytesToBeSent);
    }

    void didReceiveResponse(ScriptExecutionContextIdentifier mainContext, std::optional<ResourceLoaderIdentifier> identifier, const ResourceResponse& response)
    {
        if (m_client)
            m_client->didReceiveResponse(mainContext, identifier, response);
    }

    void didReceiveData(const SharedBuffer& buffer)
    {
        if (m_client)
            m_client->didReceiveData(buffer);
    }

    void didFinishLoading(ScriptExecutionContextIdentifier mainContext, std::optional<ResourceLoaderIdentifier> identifier, const NetworkLoadMetrics& metrics)
    {
        m_done = true;
        if (m_client)
            m_client->didFinishLoading(mainContext, identifier, metrics);
    }

    void notifyIsDone(bool isDone)
    {
        if (m_client)
            m_client->notifyIsDone(isDone);
    }

    void didFail(std::optional<ScriptExecutionContextIdentifier> mainContext, const ResourceError& error)
    {
        m_done = true;
        if (m_client)
            m_client->didFail(mainContext, error);
    }

    const String& initiator() const { return m_initiator; }

protected:
    explicit ThreadableLoaderClientWrapper(ThreadableLoaderClient&, const String&);

    WeakPtr<ThreadableLoaderClient> m_client;
    String m_initiator;
    bool m_done { false };
};

inline ThreadableLoaderClientWrapper::ThreadableLoaderClientWrapper(ThreadableLoaderClient& client, const String& initiator)
    : m_client(client)
    , m_initiator(initiator.isolatedCopy())
{
}

} // namespace WebCore
