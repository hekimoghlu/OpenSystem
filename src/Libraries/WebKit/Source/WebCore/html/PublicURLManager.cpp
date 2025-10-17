/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
#include "PublicURLManager.h"

#include "ContextDestructionObserverInlines.h"
#include "SecurityOrigin.h"
#include "URLRegistry.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PublicURLManager);

Ref<PublicURLManager> PublicURLManager::create(ScriptExecutionContext* context)
{
    Ref publicURLManager = adoptRef(*new PublicURLManager(context));
    publicURLManager->suspendIfNeeded();
    return publicURLManager;
}

PublicURLManager::PublicURLManager(ScriptExecutionContext* context)
    : ActiveDOMObject(context)
{
}

void PublicURLManager::registerURL(const URL& url, URLRegistrable& registrable)
{
    if (m_isStopped || !scriptExecutionContext())
        return;

    registrable.registry().registerURL(*scriptExecutionContext(), url, registrable);
}

void PublicURLManager::revoke(const URL& url)
{
    if (m_isStopped || !scriptExecutionContext())
        return;

    RefPtr contextOrigin = scriptExecutionContext()->securityOrigin();
    if (!contextOrigin)
        return;

    auto urlOrigin = SecurityOrigin::create(url);
    if (!urlOrigin->isSameOriginAs(*contextOrigin))
        return;

    URLRegistry::forEach([&](auto& registry) {
        registry.unregisterURL(url, scriptExecutionContext()->topOrigin().data());
    });
}

void PublicURLManager::stop()
{
    if (m_isStopped)
        return;

    m_isStopped = true;
    if (RefPtr context = scriptExecutionContext()) {
        URLRegistry::forEach([&](auto& registry) {
            registry.unregisterURLsForContext(*context);
        });
    }
}

} // namespace WebCore
