/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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
#import "APIWebArchiveResource.h"

#if PLATFORM(COCOA)

#import "APIData.h"
#import <WebCore/ArchiveResource.h>
#import <wtf/RetainPtr.h>
#import <wtf/URL.h>
#import <wtf/cf/VectorCF.h>
#import <wtf/text/WTFString.h>

namespace API {
using namespace WebCore;

Ref<WebArchiveResource> WebArchiveResource::create(API::Data* data, const WTF::String& url, const WTF::String& mimeType, const WTF::String& textEncoding)
{
    return adoptRef(*new WebArchiveResource(data, url, mimeType, textEncoding));
}

Ref<WebArchiveResource> WebArchiveResource::create(RefPtr<ArchiveResource>&& archiveResource)
{
    return adoptRef(*new WebArchiveResource(WTFMove(archiveResource)));
}

WebArchiveResource::WebArchiveResource(API::Data* data, const WTF::String& url, const WTF::String& mimeType, const WTF::String& textEncoding)
    : m_archiveResource(ArchiveResource::create(SharedBuffer::create(data->span()), WTF::URL { url }, mimeType, textEncoding, WTF::String()))
{
}

WebArchiveResource::WebArchiveResource(RefPtr<ArchiveResource>&& archiveResource)
    : m_archiveResource(WTFMove(archiveResource))
{
}

WebArchiveResource::~WebArchiveResource() = default;

Ref<API::Data> WebArchiveResource::data()
{
    RetainPtr cfData = m_archiveResource->data().makeContiguous()->createCFData();
    auto cfDataSpan = span(cfData.get());
    return API::Data::createWithoutCopying(cfDataSpan, [cfData = WTFMove(cfData)] { });
}

WTF::String WebArchiveResource::url()
{
    return m_archiveResource->url().string();
}

WTF::String WebArchiveResource::mimeType()
{
    return m_archiveResource->mimeType();
}

WTF::String WebArchiveResource::textEncoding()
{
    return m_archiveResource->textEncoding();
}

ArchiveResource* WebArchiveResource::coreArchiveResource()
{
    return m_archiveResource.get();
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
