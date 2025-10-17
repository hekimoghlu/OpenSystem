/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#include "MediaSourceRegistry.h"

#if ENABLE(MEDIA_SOURCE)

#include "MediaSource.h"
#include "ScriptExecutionContext.h"
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/URL.h>

namespace WebCore {

MediaSourceRegistry& MediaSourceRegistry::registry()
{
    ASSERT(isMainThread());
    static NeverDestroyed<MediaSourceRegistry> instance;
    return instance;
}

void MediaSourceRegistry::registerURL(const ScriptExecutionContext& context, const URL& url, URLRegistrable& registrable)
{
    ASSERT(&registrable.registry() == this);
    ASSERT(isMainThread());

    auto& urlString = url.string();
    m_urlsPerContext.add(context.identifier(), HashSet<String>()).iterator->value.add(urlString);

    Ref source = downcast<MediaSource>(registrable);
    source->addedToRegistry();
    m_mediaSources.add(urlString, std::pair { WTFMove(source), context.identifier() });
}

void MediaSourceRegistry::unregisterURL(const URL& url, const SecurityOriginData&)
{
    // MediaSource objects are not exposed to workers.
    if (!isMainThread())
        return;

    auto& urlString = url.string();
    auto [source, contextIdentifier] = m_mediaSources.take(urlString);
    if (!source)
        return;

    source->removedFromRegistry();

    auto m_urlsPerContextIterator = m_urlsPerContext.find(contextIdentifier);
    ASSERT(m_urlsPerContextIterator != m_urlsPerContext.end());
    ASSERT(m_urlsPerContextIterator->value.contains(urlString));
    m_urlsPerContextIterator->value.remove(urlString);
    if (m_urlsPerContextIterator->value.isEmpty())
        m_urlsPerContext.remove(m_urlsPerContextIterator);
}

void MediaSourceRegistry::unregisterURLsForContext(const ScriptExecutionContext& context)
{
    // MediaSource objects are not exposed to workers.
    if (!isMainThread())
        return;

    auto urls = m_urlsPerContext.take(context.identifier());
    for (auto& url : urls) {
        ASSERT(m_mediaSources.contains(url));
        auto [source, contextIdentifier] = m_mediaSources.take(url);
        source->removedFromRegistry();
    }
}

URLRegistrable* MediaSourceRegistry::lookup(const String& url) const
{
    ASSERT(isMainThread());
    if (auto it = m_mediaSources.find(url); it != m_mediaSources.end())
        return it->value.first.get();
    return nullptr;
}

MediaSourceRegistry::MediaSourceRegistry()
{
    MediaSource::setRegistry(this);
}

} // namespace WebCore

#endif
