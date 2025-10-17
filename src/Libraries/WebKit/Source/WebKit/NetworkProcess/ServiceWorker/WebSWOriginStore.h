/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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

#include "SharedStringHashStore.h"
#include <WebCore/SWOriginStore.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>

namespace WebKit {

class WebSWServerConnection;

class WebSWOriginStore final : public WebCore::SWOriginStore, private SharedStringHashStore::Client {
    WTF_MAKE_TZONE_ALLOCATED(WebSWOriginStore);
public:
    WebSWOriginStore();

    void registerSWServerConnection(WebSWServerConnection&);
    void unregisterSWServerConnection(WebSWServerConnection&);
    void importComplete() final;

private:
    void sendStoreHandle(WebSWServerConnection&);

    void addToStore(const WebCore::SecurityOriginData&) final;
    void removeFromStore(const WebCore::SecurityOriginData&) final;
    void clearStore() final;

    // SharedStringHashStore::Client.
    void didInvalidateSharedMemory() final;

    SharedStringHashStore m_store;
    bool m_isImported { false };
    WeakHashSet<WebSWServerConnection> m_webSWServerConnections;
};

} // namespace WebKit
