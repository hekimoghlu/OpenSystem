/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
#include "WebDatabaseProvider.h"

#include "NetworkProcessConnection.h"
#include "WebIDBConnectionToServer.h"
#include "WebProcess.h"
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {
using namespace WebCore;

static HashMap<PageGroupIdentifier, WeakRef<WebDatabaseProvider>>& databaseProviders()
{
    static NeverDestroyed<HashMap<PageGroupIdentifier, WeakRef<WebDatabaseProvider>>> databaseProviders;

    return databaseProviders;
}

Ref<WebDatabaseProvider> WebDatabaseProvider::getOrCreate(PageGroupIdentifier identifier)
{
    RefPtr<WebDatabaseProvider> databaseProvider;
    auto& slot = databaseProviders().ensure(identifier, [&] {
        databaseProvider = adoptRef(new WebDatabaseProvider(identifier));
        return WeakRef { *databaseProvider };
    }).iterator->value;
    return databaseProvider ? databaseProvider.releaseNonNull() : Ref { slot.get() };
}

WebDatabaseProvider::WebDatabaseProvider(PageGroupIdentifier identifier)
    : m_identifier(identifier)
{
}

WebDatabaseProvider::~WebDatabaseProvider()
{
    ASSERT(databaseProviders().contains(m_identifier));

    databaseProviders().remove(m_identifier);
}

WebCore::IDBClient::IDBConnectionToServer& WebDatabaseProvider::idbConnectionToServerForSession(PAL::SessionID)
{
    return WebProcess::singleton().ensureProtectedNetworkProcessConnection()->idbConnectionToServer().coreConnectionToServer();
}

}
