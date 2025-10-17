/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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

#include "APIObject.h"
#include "DataTaskIdentifier.h"
#include <pal/SessionID.h>
#include <wtf/URL.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class ResourceError;
}

namespace WebKit {
class NetworkProcessProxy;
class ProcessThrottlerActivity;
class WebPageProxy;
}

namespace API {

class DataTaskClient;

class DataTask : public API::ObjectImpl<API::Object::Type::DataTask> {
public:

    template<typename... Args> static Ref<DataTask> create(Args&&... args)
    {
        return adoptRef(*new DataTask(std::forward<Args>(args)...));
    }
    ~DataTask();

    void cancel();

    WebKit::WebPageProxy* page() { return m_page.get(); }
    const WTF::URL& originalURL() const { return m_originalURL; }
    const DataTaskClient& client() const { return m_client.get(); }
    Ref<DataTaskClient> protectedClient() const;
    void setClient(Ref<DataTaskClient>&&);
    void networkProcessCrashed();
    void didCompleteWithError(WebCore::ResourceError&&);

private:
    DataTask(std::optional<WebKit::DataTaskIdentifier>, WeakPtr<WebKit::WebPageProxy>&&, WTF::URL&&, bool shouldRunAtForegroundPriority);

    Markable<WebKit::DataTaskIdentifier> m_identifier;
    WeakPtr<WebKit::WebPageProxy> m_page;
    WTF::URL m_originalURL;
    WeakPtr<WebKit::NetworkProcessProxy> m_networkProcess;
    std::optional<PAL::SessionID> m_sessionID;
    Ref<DataTaskClient> m_client;
    RefPtr<WebKit::ProcessThrottlerActivity> m_activity;
};

} // namespace API
