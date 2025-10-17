/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
#import "config.h"
#import "APIWebArchive.h"

#if PLATFORM(COCOA)

#import "APIArray.h"
#import "APIData.h"
#import "APIWebArchiveResource.h"
#import <WebCore/LegacyWebArchive.h>
#import <wtf/RetainPtr.h>
#import <wtf/cf/VectorCF.h>

namespace API {
using namespace WebCore;

Ref<WebArchive> WebArchive::create(WebArchiveResource* mainResource, RefPtr<API::Array>&& subresources, RefPtr<API::Array>&& subframeArchives)
{
    return adoptRef(*new WebArchive(mainResource, WTFMove(subresources), WTFMove(subframeArchives)));
}

Ref<WebArchive> WebArchive::create(API::Data* data)
{
    return adoptRef(*new WebArchive(data));
}

Ref<WebArchive> WebArchive::create(RefPtr<LegacyWebArchive>&& legacyWebArchive)
{
    return adoptRef(*new WebArchive(legacyWebArchive.releaseNonNull()));
}

Ref<WebArchive> WebArchive::create(const SimpleRange& range)
{
    return adoptRef(*new WebArchive(LegacyWebArchive::create(range)));
}

WebArchive::WebArchive(WebArchiveResource* mainResource, RefPtr<API::Array>&& subresources, RefPtr<API::Array>&& subframeArchives)
    : m_cachedMainResource(mainResource)
    , m_cachedSubresources(subresources)
    , m_cachedSubframeArchives(subframeArchives)
{
    auto coreMainResource = m_cachedMainResource->coreArchiveResource();

    Vector<Ref<ArchiveResource>> coreArchiveResources(m_cachedSubresources->size(), [&](size_t i) {
        RefPtr resource = m_cachedSubresources->at<WebArchiveResource>(i);
        ASSERT(resource);
        ASSERT(resource->coreArchiveResource());
        return Ref<ArchiveResource> { *resource->coreArchiveResource() };
    });

    Vector<Ref<LegacyWebArchive>> coreSubframeLegacyWebArchives(m_cachedSubframeArchives->size(), [&](size_t i) {
        RefPtr subframeWebArchive = m_cachedSubframeArchives->at<WebArchive>(i);
        ASSERT(subframeWebArchive);
        ASSERT(subframeWebArchive->coreLegacyWebArchive());
        return Ref<LegacyWebArchive> { *subframeWebArchive->coreLegacyWebArchive() };
    });

    m_legacyWebArchive = LegacyWebArchive::create(*coreMainResource, WTFMove(coreArchiveResources), WTFMove(coreSubframeLegacyWebArchives));
}

WebArchive::WebArchive(API::Data* data)
{
    m_legacyWebArchive = LegacyWebArchive::create(SharedBuffer::create(data->span()).get());
}

WebArchive::WebArchive(RefPtr<LegacyWebArchive>&& legacyWebArchive)
    : m_legacyWebArchive(legacyWebArchive)
{
}

WebArchive::~WebArchive()
{
}

WebArchiveResource* WebArchive::mainResource()
{
    if (!m_cachedMainResource)
        m_cachedMainResource = WebArchiveResource::create(m_legacyWebArchive->mainResource());
    return m_cachedMainResource.get();
}

API::Array* WebArchive::subresources()
{
    if (!m_cachedSubresources) {
        auto subresources = WTF::map(m_legacyWebArchive->subresources(), [](auto& subresource) -> RefPtr<API::Object> {
            return WebArchiveResource::create(subresource.ptr());
        });
        m_cachedSubresources = API::Array::create(WTFMove(subresources));
    }

    return m_cachedSubresources.get();
}

API::Array* WebArchive::subframeArchives()
{
    if (!m_cachedSubframeArchives) {
        auto subframeWebArchives = WTF::map(m_legacyWebArchive->subframeArchives(), [](auto& subframeArchive) -> RefPtr<API::Object> {
            return WebArchive::create(static_cast<LegacyWebArchive*>(subframeArchive.ptr()));
        });
        m_cachedSubframeArchives = API::Array::create(WTFMove(subframeWebArchives));
    }

    return m_cachedSubframeArchives.get();
}

Ref<API::Data> WebArchive::data()
{
    RetainPtr rawDataRepresentation = m_legacyWebArchive->rawDataRepresentation();
    auto rawDataSpan = span(rawDataRepresentation.get());
    return API::Data::createWithoutCopying(rawDataSpan, [rawDataRepresentation = WTFMove(rawDataRepresentation)] { });
}

LegacyWebArchive* WebArchive::coreLegacyWebArchive()
{
    return m_legacyWebArchive.get();
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
