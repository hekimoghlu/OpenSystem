/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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
#include "ArchiveResourceCollection.h"

#include "Archive.h"

namespace WebCore {

void ArchiveResourceCollection::addAllResources(Archive& archive)
{
    for (auto& subresource : archive.subresources())
        m_subresources.set(subresource->url().string(), subresource.ptr());

    for (auto& subframeArchive : archive.subframeArchives()) {
        ASSERT(subframeArchive->mainResource());
        auto frameName = subframeArchive->mainResource()->frameName();
        if (frameName.isNull()) {
            // In the MHTML case, frames don't have a name, so we use the URL instead.
            frameName = subframeArchive->mainResource()->url().string();
        }
        m_subframes.set(frameName, subframeArchive.ptr());
    }
}

// FIXME: Adding a resource directly to a DocumentLoader/ArchiveResourceCollection seems like bad design, but is API some apps rely on.
// Can we change the design in a manner that will let us deprecate that API without reducing functionality of those apps?
void ArchiveResourceCollection::addResource(Ref<ArchiveResource>&& resource)
{
    auto& url = resource->url();
    m_subresources.set(url.string(), WTFMove(resource));
}

ArchiveResource* ArchiveResourceCollection::archiveResourceForURL(const URL& url)
{
    if (RefPtr resource = m_subresources.get(url.string()))
        return resource.get();
    if (!url.protocolIs("https"_s))
        return nullptr;
    URL httpURL = url;
    httpURL.setProtocol("http"_s);
    return m_subresources.get(httpURL.string());
}

RefPtr<Archive> ArchiveResourceCollection::popSubframeArchive(const String& frameName, const URL& url)
{
    if (auto archive = m_subframes.take(frameName))
        return archive;
    return m_subframes.take(url.string());
}

}
