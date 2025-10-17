/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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
#include "MediaResourceSniffer.h"

#if ENABLE(VIDEO)

#include "MIMESniffer.h"
#include "ResourceRequest.h"
#include <limits.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaResourceSniffer);

Ref<MediaResourceSniffer> MediaResourceSniffer::create(PlatformMediaResourceLoader& loader, ResourceRequest&& request, std::optional<size_t> maxSize)
{
    if (maxSize)
        request.addHTTPHeaderField(HTTPHeaderName::Range, makeString("bytes="_s, 0, '-', *maxSize));
    auto resource = loader.requestResource(WTFMove(request), PlatformMediaResourceLoader::LoadOption::DisallowCaching);
    if (!resource)
        return adoptRef(*new MediaResourceSniffer());
    Ref sniffer = adoptRef(*new MediaResourceSniffer(*resource , maxSize.value_or(SIZE_MAX)));
    resource->setClient(sniffer.copyRef());
    return sniffer;
}

MediaResourceSniffer::MediaResourceSniffer()
    : m_maxSize(0)
{
    m_producer.reject(PlatformMediaError::NetworkError);
}

MediaResourceSniffer::MediaResourceSniffer(Ref<PlatformMediaResource>&& resource, size_t maxSize)
    : m_resource(WTFMove(resource))
    , m_maxSize(maxSize)
{
}

MediaResourceSniffer::~MediaResourceSniffer()
{
    cancel();
}

void MediaResourceSniffer::cancel()
{
    if (auto resource = std::exchange(m_resource, { }))
        resource->shutdown();
    if (!m_producer.isSettled())
        m_producer.reject(PlatformMediaError::Cancelled);
    m_content.reset();
}

MediaResourceSniffer::Promise& MediaResourceSniffer::promise() const
{
    return m_producer.promise().get();
}

void MediaResourceSniffer::dataReceived(PlatformMediaResource&, const SharedBuffer& buffer)
{
    m_received += buffer.size();
    m_content.append(buffer);
    auto contiguousBuffer = m_content.get()->makeContiguous();
    auto mimeType = MIMESniffer::getMIMETypeFromContent(contiguousBuffer->span());
    if (mimeType.isEmpty() && m_received < m_maxSize)
        return;
    if (!m_producer.isSettled())
        m_producer.resolve(ContentType { WTFMove(mimeType) });
    cancel();
}

void MediaResourceSniffer::loadFailed(PlatformMediaResource&, const ResourceError&)
{
    if (!m_producer.isSettled())
        m_producer.reject(PlatformMediaError::NetworkError);
    cancel();
}

void MediaResourceSniffer::loadFinished(PlatformMediaResource&, const NetworkLoadMetrics&)
{
    if (m_producer.isSettled())
        return;
    Ref contiguousBuffer = m_content.takeAsContiguous();
    auto mimeType = MIMESniffer::getMIMETypeFromContent(contiguousBuffer->span());
    m_producer.resolve(ContentType { WTFMove(mimeType) });
    cancel();
}

} // namespace WebCore

#endif // ENABLE(VIDEO)
