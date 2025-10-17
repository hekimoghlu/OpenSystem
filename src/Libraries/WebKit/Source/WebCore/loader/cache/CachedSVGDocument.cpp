/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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
#include "CachedSVGDocument.h"

#include "ParserContentPolicy.h"
#include "Settings.h"
#include "SharedBuffer.h"

namespace WebCore {

CachedSVGDocument::CachedSVGDocument(CachedResourceRequest&& request, PAL::SessionID sessionID, const CookieJar* cookieJar, const Settings& settings)
    : CachedResource(WTFMove(request), Type::SVGDocumentResource, sessionID, cookieJar)
    , m_decoder(TextResourceDecoder::create("application/xml"_s))
    , m_settings(settings)
{
}

CachedSVGDocument::CachedSVGDocument(CachedResourceRequest&& request, CachedSVGDocument& resource)
    : CachedSVGDocument(WTFMove(request), resource.sessionID(), resource.cookieJar(), resource.m_settings)
{
}

CachedSVGDocument::~CachedSVGDocument() = default;

void CachedSVGDocument::setEncoding(const String& chs)
{
    protectedDecoder()->setEncoding(chs, TextResourceDecoder::EncodingFromHTTPHeader);
}

ASCIILiteral CachedSVGDocument::encoding() const
{
    return protectedDecoder()->encoding().name();
}

RefPtr<TextResourceDecoder> CachedSVGDocument::protectedDecoder() const
{
    return m_decoder;
}

void CachedSVGDocument::finishLoading(const FragmentedSharedBuffer* data, const NetworkLoadMetrics& metrics)
{
    if (data) {
        // We don't need to create a new frame because the new document belongs to the parent UseElement.
        Ref document = SVGDocument::create(nullptr, m_settings.copyRef(), response().url());
        document->setMarkupUnsafe(protectedDecoder()->decodeAndFlush(data->makeContiguous()->span()), { ParserContentPolicy::AllowDeclarativeShadowRoots });
        m_document = WTFMove(document);
    }
    CachedResource::finishLoading(data, metrics);
}

}
