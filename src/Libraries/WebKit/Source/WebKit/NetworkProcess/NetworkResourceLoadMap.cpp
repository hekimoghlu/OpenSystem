/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
#include "NetworkResourceLoadMap.h"
#include "NetworkResourceLoader.h"

namespace WebKit {

NetworkResourceLoadMap::NetworkResourceLoadMap(Function<void(bool hasUpload)>&& hasUploadChangeListener)
    : m_hasUploadChangeListener(WTFMove(hasUploadChangeListener))
{
}

NetworkResourceLoadMap::~NetworkResourceLoadMap()
{
    clear();
}

NetworkResourceLoadMap::MapType::AddResult NetworkResourceLoadMap::add(WebCore::ResourceLoaderIdentifier identifier, Ref<NetworkResourceLoader>&& loader)
{
    ASSERT(!m_loaders.contains(identifier));
    bool hasUpload = loader->originalRequest().hasUpload();
    auto result = m_loaders.add(identifier, WTFMove(loader));
    if (hasUpload)
        setHasUpload(true);
    return result;
}

bool NetworkResourceLoadMap::remove(WebCore::ResourceLoaderIdentifier identifier)
{
    return !!take(identifier);
}

void NetworkResourceLoadMap::clear()
{
    m_loaders.clear();
    setHasUpload(false);
}

RefPtr<NetworkResourceLoader> NetworkResourceLoadMap::take(WebCore::ResourceLoaderIdentifier identifier)
{
    auto loader = m_loaders.take(identifier);
    if (!loader)
        return nullptr;

    if (loader->originalRequest().hasUpload())
        setHasUpload(WTF::anyOf(m_loaders.values(), [](auto& loader) { return loader->originalRequest().hasUpload(); }));

    return loader;
}

NetworkResourceLoader* NetworkResourceLoadMap::get(WebCore::ResourceLoaderIdentifier identifier) const
{
    return m_loaders.get(identifier);
}

void NetworkResourceLoadMap::setHasUpload(bool hasUpload)
{
    if (m_hasUpload == hasUpload)
        return;

    m_hasUpload = hasUpload;
    if (m_hasUploadChangeListener)
        m_hasUploadChangeListener(m_hasUpload);
}

} // namespace WebKit
