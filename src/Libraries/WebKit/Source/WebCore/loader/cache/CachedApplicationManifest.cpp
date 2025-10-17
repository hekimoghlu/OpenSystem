/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
#include "CachedApplicationManifest.h"

#if ENABLE(APPLICATION_MANIFEST)

#include "ApplicationManifestParser.h"
#include "SharedBuffer.h"
#include "TextResourceDecoder.h"

namespace WebCore {

CachedApplicationManifest::CachedApplicationManifest(CachedResourceRequest&& request, PAL::SessionID sessionID, const CookieJar* cookieJar)
    : CachedResource(WTFMove(request), Type::ApplicationManifest, sessionID, cookieJar)
    , m_decoder(TextResourceDecoder::create("application/manifest+json"_s, PAL::UTF8Encoding()))
{
}

void CachedApplicationManifest::finishLoading(const FragmentedSharedBuffer* data, const NetworkLoadMetrics& metrics)
{
    if (data) {
        Ref contiguousData = data->makeContiguous();
        setEncodedSize(data->size());
        m_text = protectedDecoder()->decodeAndFlush(contiguousData->span());
        m_data = WTFMove(contiguousData);
    } else {
        m_data = nullptr;
        setEncodedSize(0);
    }
    CachedResource::finishLoading(data, metrics);
}

void CachedApplicationManifest::setEncoding(const String& chs)
{
    protectedDecoder()->setEncoding(chs, TextResourceDecoder::EncodingFromHTTPHeader);
}

ASCIILiteral CachedApplicationManifest::encoding() const
{
    return protectedDecoder()->encoding().name();
}

std::optional<ApplicationManifest> CachedApplicationManifest::process(const URL& manifestURL, const URL& documentURL, Document* document)
{
    if (!m_text)
        return std::nullopt;
    if (document)
        return ApplicationManifestParser::parse(*document, *m_text, manifestURL, documentURL);
    return ApplicationManifestParser::parse(*m_text, manifestURL, documentURL);
}

Ref<TextResourceDecoder> CachedApplicationManifest::protectedDecoder() const
{
    return m_decoder;
}

} // namespace WebCore

#endif // ENABLE(APPLICATION_MANIFEST)
