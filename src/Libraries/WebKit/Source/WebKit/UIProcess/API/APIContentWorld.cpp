/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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
#include "APIContentWorld.h"

#include "ContentWorldShared.h"
#include "WebUserContentControllerProxy.h"
#include <wtf/HashMap.h>
#include <wtf/WeakRef.h>
#include <wtf/text/StringHash.h>

namespace API {

static HashMap<WTF::String, WeakRef<ContentWorld>>& sharedWorldNameMap()
{
    static NeverDestroyed<HashMap<WTF::String, WeakRef<ContentWorld>>> sharedMap;
    return sharedMap;
}

static HashMap<WebKit::ContentWorldIdentifier, WeakRef<ContentWorld>>& sharedWorldIdentifierMap()
{
    static NeverDestroyed<HashMap<WebKit::ContentWorldIdentifier, WeakRef<ContentWorld>>> sharedMap;
    return sharedMap;
}

ContentWorld* ContentWorld::worldForIdentifier(WebKit::ContentWorldIdentifier identifier)
{
    return sharedWorldIdentifierMap().get(identifier);
}

static WebKit::ContentWorldIdentifier generateIdentifier()
{
    static std::once_flag once;
    std::call_once(once, [] {
        // To make sure we don't use our shared pageContentWorld identifier for this
        // content world we're about to make, burn through one identifier.
        auto identifier = WebKit::ContentWorldIdentifier::generate();
        ASSERT_UNUSED(identifier, identifier.toUInt64() >= WebKit::pageContentWorldIdentifier().toUInt64());
    });
    return WebKit::ContentWorldIdentifier::generate();
}

ContentWorld::ContentWorld(const WTF::String& name, OptionSet<WebKit::ContentWorldOption> options)
    : m_identifier(generateIdentifier())
    , m_name(name)
    , m_options(options)
{
    auto addResult = sharedWorldIdentifierMap().add(m_identifier, *this);
    ASSERT_UNUSED(addResult, addResult.isNewEntry);
}

ContentWorld::ContentWorld(WebKit::ContentWorldIdentifier identifier)
    : m_identifier(identifier)
{
    ASSERT(m_identifier == WebKit::pageContentWorldIdentifier());
}

Ref<ContentWorld> ContentWorld::sharedWorldWithName(const WTF::String& name, OptionSet<WebKit::ContentWorldOption> options)
{
    RefPtr<ContentWorld> newContentWorld;
    auto result = sharedWorldNameMap().ensure(name, [&] {
        newContentWorld = adoptRef(*new ContentWorld(name, options));
        return WeakRef { *newContentWorld };
    });
    return newContentWorld ? newContentWorld.releaseNonNull() : Ref { result.iterator->value.get() };
}

ContentWorld& ContentWorld::pageContentWorldSingleton()
{
    static NeverDestroyed<Ref<ContentWorld>> world(adoptRef(*new ContentWorld(WebKit::pageContentWorldIdentifier())));
    return world.get();
}

ContentWorld& ContentWorld::defaultClientWorldSingleton()
{
    static NeverDestroyed<Ref<ContentWorld>> world(adoptRef(*new ContentWorld(WTF::String { }, { })));
    return world.get();
}

ContentWorld::~ContentWorld()
{
    ASSERT(m_identifier != WebKit::pageContentWorldIdentifier());

    auto result = sharedWorldIdentifierMap().take(m_identifier);
    ASSERT_UNUSED(result, result.get() == this || m_identifier == WebKit::pageContentWorldIdentifier());

    if (!name().isNull()) {
        auto taken = sharedWorldNameMap().take(name());
        ASSERT_UNUSED(taken, taken.get() == this);
    }

    for (Ref proxy : m_associatedContentControllerProxies)
        proxy->contentWorldDestroyed(*this);
}

void ContentWorld::addAssociatedUserContentControllerProxy(WebKit::WebUserContentControllerProxy& proxy)
{
    auto addResult = m_associatedContentControllerProxies.add(proxy);
    ASSERT_UNUSED(addResult, addResult.isNewEntry);
}

void ContentWorld::userContentControllerProxyDestroyed(WebKit::WebUserContentControllerProxy& proxy)
{
    bool removed = m_associatedContentControllerProxies.remove(proxy);
    ASSERT_UNUSED(removed, removed);
}

} // namespace API
