/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
#include "ApplicationCacheResource.h"
#include <stdio.h>

namespace WebCore {

Ref<ApplicationCacheResource> ApplicationCacheResource::create(const URL& url, const ResourceResponse& response, unsigned type, RefPtr<FragmentedSharedBuffer>&& buffer, const String& path)
{
    ASSERT(!url.hasFragmentIdentifier());
    if (!buffer)
        buffer = SharedBuffer::create();
    auto resourceResponse = response;
    resourceResponse.setSource(ResourceResponse::Source::ApplicationCache);

    return adoptRef(*new ApplicationCacheResource(URL { url }, WTFMove(resourceResponse), type, buffer.releaseNonNull(), path));
}

ApplicationCacheResource::ApplicationCacheResource(URL&& url, ResourceResponse&& response, unsigned type, Ref<FragmentedSharedBuffer>&& data, const String& path)
    : SubstituteResource(WTFMove(url), WTFMove(response), WTFMove(data))
    , m_type(type)
    , m_storageID(0)
    , m_estimatedSizeInStorage(0)
    , m_path(path)
{
}

void ApplicationCacheResource::deliver(ResourceLoader& loader)
{
    if (m_path.isEmpty())
        loader.deliverResponseAndData(response(), RefPtr { &data() });
    else
        loader.deliverResponseAndData(response(), SharedBuffer::createWithContentsOfFile(m_path));
}

void ApplicationCacheResource::addType(unsigned type) 
{
    // Caller should take care of storing the new type in database.
    m_type |= type; 
}

int64_t ApplicationCacheResource::estimatedSizeInStorage()
{
    if (m_estimatedSizeInStorage)
      return m_estimatedSizeInStorage;

    m_estimatedSizeInStorage = data().size();

    for (const auto& headerField : response().httpHeaderFields())
        m_estimatedSizeInStorage += (headerField.key.length() + headerField.value.length() + 2) * sizeof(UChar);

    m_estimatedSizeInStorage += url().string().length() * sizeof(UChar);
    m_estimatedSizeInStorage += sizeof(int); // response().m_httpStatusCode
    m_estimatedSizeInStorage += response().url().string().length() * sizeof(UChar);
    m_estimatedSizeInStorage += sizeof(unsigned); // dataId
    m_estimatedSizeInStorage += response().mimeType().length() * sizeof(UChar);
    m_estimatedSizeInStorage += response().textEncodingName().length() * sizeof(UChar);

    return m_estimatedSizeInStorage;
}

#ifndef NDEBUG
void ApplicationCacheResource::dumpType(unsigned type)
{
    if (type & Master)
        printf("master ");
    if (type & Manifest)
        printf("manifest ");
    if (type & Explicit)
        printf("explicit ");
    if (type & Foreign)
        printf("foreign ");
    if (type & Fallback)
        printf("fallback ");
    
    printf("\n");
}
#endif

} // namespace WebCore
