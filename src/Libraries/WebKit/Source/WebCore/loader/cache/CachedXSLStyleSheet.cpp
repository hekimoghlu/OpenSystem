/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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
#include "CachedXSLStyleSheet.h"

#include "CachedResourceClientWalker.h"
#include "CachedStyleSheetClient.h"
#include "SharedBuffer.h"
#include "TextResourceDecoder.h"

namespace WebCore {

#if ENABLE(XSLT)

CachedXSLStyleSheet::CachedXSLStyleSheet(CachedResourceRequest&& request, PAL::SessionID sessionID, const CookieJar* cookieJar)
    : CachedResource(WTFMove(request), Type::XSLStyleSheet, sessionID, cookieJar)
    , m_decoder(TextResourceDecoder::create("text/xsl"_s))
{
}

CachedXSLStyleSheet::~CachedXSLStyleSheet() = default;

RefPtr<TextResourceDecoder> CachedXSLStyleSheet::protectedDecoder() const
{
    return m_decoder;
}

void CachedXSLStyleSheet::didAddClient(CachedResourceClient& client)
{
    ASSERT(client.resourceClientType() == CachedStyleSheetClient::expectedType());
    if (!isLoading())
        downcast<CachedStyleSheetClient>(client).setXSLStyleSheet(m_resourceRequest.url().string(), response().url(), m_sheet);
}

void CachedXSLStyleSheet::setEncoding(const String& chs)
{
    protectedDecoder()->setEncoding(chs, TextResourceDecoder::EncodingFromHTTPHeader);
}

ASCIILiteral CachedXSLStyleSheet::encoding() const
{
    return protectedDecoder()->encoding().name();
}

void CachedXSLStyleSheet::finishLoading(const FragmentedSharedBuffer* data, const NetworkLoadMetrics& metrics)
{
    if (data) {
        Ref contiguousData = data->makeContiguous();
        setEncodedSize(data->size());
        m_sheet = protectedDecoder()->decodeAndFlush(contiguousData->span());
        m_data = WTFMove(contiguousData);
    } else {
        m_data = nullptr;
        setEncodedSize(0);
    }
    setLoading(false);
    checkNotify(metrics);
}

void CachedXSLStyleSheet::checkNotify(const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess)
{
    if (isLoading())
        return;
    
    CachedResourceClientWalker<CachedStyleSheetClient> walker(*this);
    while (CachedStyleSheetClient* c = walker.next())
        c->setXSLStyleSheet(m_resourceRequest.url().string(), response().url(), m_sheet);
}

#endif

}
