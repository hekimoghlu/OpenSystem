/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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

#include <WebCore/ResourceLoaderIdentifier.h>
#include <wtf/Function.h>
#include <wtf/HashMap.h>

namespace WebKit {

class NetworkResourceLoader;
class NetworkConnectionToWebProcess;

class NetworkResourceLoadMap {
public:
    using MapType = HashMap<WebCore::ResourceLoaderIdentifier, Ref<NetworkResourceLoader>>;
    NetworkResourceLoadMap(Function<void(bool hasUpload)>&&);
    ~NetworkResourceLoadMap();

    bool isEmpty() const { return m_loaders.isEmpty(); }
    bool contains(WebCore::ResourceLoaderIdentifier identifier) const { return m_loaders.contains(identifier); }
    MapType::iterator begin() { return m_loaders.begin(); }
    MapType::ValuesIteratorRange values() { return m_loaders.values(); }
    void clear();

    MapType::AddResult add(WebCore::ResourceLoaderIdentifier, Ref<NetworkResourceLoader>&&);
    NetworkResourceLoader* get(WebCore::ResourceLoaderIdentifier) const;
    bool remove(WebCore::ResourceLoaderIdentifier);
    RefPtr<NetworkResourceLoader> take(WebCore::ResourceLoaderIdentifier);

    bool hasUpload() const { return m_hasUpload; }

private:
    void setHasUpload(bool);

    MapType m_loaders;
    bool m_hasUpload { false };
    Function<void(bool hasUpload)> m_hasUploadChangeListener;
};

} // namespace WebKit
