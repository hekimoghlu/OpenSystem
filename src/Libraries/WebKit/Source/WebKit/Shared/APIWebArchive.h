/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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

#if PLATFORM(COCOA)

#include "APIObject.h"
#include <wtf/RefPtr.h>

namespace API {
class Array;
class Data;
}

namespace WebCore {
class LegacyWebArchive;
struct SimpleRange;
}

namespace API {

class WebArchiveResource;

class WebArchive : public API::ObjectImpl<API::Object::Type::WebArchive> {
public:
    virtual ~WebArchive();

    static Ref<WebArchive> create(WebArchiveResource* mainResource, RefPtr<API::Array>&& subresources, RefPtr<API::Array>&& subframeArchives);
    static Ref<WebArchive> create(API::Data*);
    static Ref<WebArchive> create(RefPtr<WebCore::LegacyWebArchive>&&);
    static Ref<WebArchive> create(const WebCore::SimpleRange&);

    WebArchiveResource* mainResource();
    API::Array* subresources();
    API::Array* subframeArchives();

    Ref<API::Data> data();

    WebCore::LegacyWebArchive* coreLegacyWebArchive();

private:
    WebArchive(WebArchiveResource* mainResource, RefPtr<API::Array>&& subresources, RefPtr<API::Array>&& subframeArchives);
    WebArchive(API::Data*);
    WebArchive(RefPtr<WebCore::LegacyWebArchive>&&);

    RefPtr<WebCore::LegacyWebArchive> m_legacyWebArchive;
    RefPtr<WebArchiveResource> m_cachedMainResource;
    RefPtr<API::Array> m_cachedSubresources;
    RefPtr<API::Array> m_cachedSubframeArchives;
};

} // namespace API

#endif // PLATFORM(COCOA)
