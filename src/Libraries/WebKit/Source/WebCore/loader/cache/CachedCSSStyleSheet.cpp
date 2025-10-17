/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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
#include "CachedCSSStyleSheet.h"

#include "CSSStyleSheet.h"
#include "CachedResourceClientWalker.h"
#include "CachedResourceRequest.h"
#include "CachedStyleSheetClient.h"
#include "CommonAtomStrings.h"
#include "HTTPHeaderNames.h"
#include "HTTPParsers.h"
#include "MemoryCache.h"
#include "ParsedContentType.h"
#include "SharedBuffer.h"
#include "StyleSheetContents.h"
#include "TextResourceDecoder.h"

#if PLATFORM(COCOA)
#include <wtf/cocoa/RuntimeApplicationChecksCocoa.h>
#endif

namespace WebCore {

CachedCSSStyleSheet::CachedCSSStyleSheet(CachedResourceRequest&& request, PAL::SessionID sessionID, const CookieJar* cookieJar)
    : CachedResource(WTFMove(request), Type::CSSStyleSheet, sessionID, cookieJar)
    , m_decoder(TextResourceDecoder::create(cssContentTypeAtom(), request.charset()))
{
}

CachedCSSStyleSheet::~CachedCSSStyleSheet()
{
    if (RefPtr parsedStyleSheetCache = m_parsedStyleSheetCache)
        parsedStyleSheetCache->removedFromMemoryCache();
}

void CachedCSSStyleSheet::didAddClient(CachedResourceClient& client)
{
    ASSERT(client.resourceClientType() == CachedStyleSheetClient::expectedType());
    // CachedResource::didAddClient() must be before setCSSStyleSheet(),
    // because setCSSStyleSheet() may cause scripts to be executed, which could destroy 'c' if it is an instance of HTMLLinkElement.
    // see the comment of HTMLLinkElement::setCSSStyleSheet.
    CachedResource::didAddClient(client);

    if (!isLoading())
        downcast<CachedStyleSheetClient>(client).setCSSStyleSheet(m_resourceRequest.url().string(), response().url(), m_decoder->encoding().name(), this);
}

void CachedCSSStyleSheet::setEncoding(const String& chs)
{
    protectedDecoder()->setEncoding(chs, TextResourceDecoder::EncodingFromHTTPHeader);
}

ASCIILiteral CachedCSSStyleSheet::encoding() const
{
    return protectedDecoder()->encoding().name();
}

const String CachedCSSStyleSheet::sheetText(MIMETypeCheckHint mimeTypeCheckHint, bool* hasValidMIMEType, bool* hasHTTPStatusOK) const
{
    // Ensure hasValidMIMEType and hasHTTPStatusOK always get set (even if m_data is null or empty) Ã¢â‚¬â€ which in turn
    // ensures that if the MIME type isn't text/css or the HTTP status isn't an OK status, we never load the resource.
    // https://html.spec.whatwg.org/#link-type-stylesheet:process-the-linked-resource
    if (!canUseSheet(mimeTypeCheckHint, hasValidMIMEType, hasHTTPStatusOK) || !m_data || m_data->isEmpty())
        return String();

    if (!m_decodedSheetText.isNull())
        return m_decodedSheetText;

    // Don't cache the decoded text, regenerating is cheap and it can use quite a bit of memory.
    return protectedDecoder()->decodeAndFlush(m_data->makeContiguous()->span());
}

void CachedCSSStyleSheet::setBodyDataFrom(const CachedResource& resource)
{
    ASSERT(resource.type() == type());
    const CachedCSSStyleSheet& sheet = static_cast<const CachedCSSStyleSheet&>(resource);

    CachedResource::setBodyDataFrom(resource);

    m_decoder = sheet.m_decoder;
    m_decodedSheetText = sheet.m_decodedSheetText;
    if (RefPtr parsedStyleSheetCache = sheet.m_parsedStyleSheetCache)
        saveParsedStyleSheet(parsedStyleSheetCache.releaseNonNull());
}

void CachedCSSStyleSheet::finishLoading(const FragmentedSharedBuffer* data, const NetworkLoadMetrics& metrics)
{
    if (data) {
        Ref contiguousData = data->makeContiguous();
        setEncodedSize(data->size());
        // Decode the data to find out the encoding and keep the sheet text around during checkNotify()
        m_decodedSheetText = protectedDecoder()->decodeAndFlush(contiguousData->span());
        m_data = WTFMove(contiguousData);
    } else {
        m_data = nullptr;
        setEncodedSize(0);
    }
    setLoading(false);
    checkNotify(metrics);
    // Clear the decoded text as it is unlikely to be needed immediately again and is cheap to regenerate.
    m_decodedSheetText = String();
}

Ref<TextResourceDecoder> CachedCSSStyleSheet::protectedDecoder() const
{
    return m_decoder;
}

void CachedCSSStyleSheet::checkNotify(const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess)
{
    if (isLoading())
        return;

    CachedResourceClientWalker<CachedStyleSheetClient> walker(*this);
    while (CachedStyleSheetClient* c = walker.next())
        c->setCSSStyleSheet(m_resourceRequest.url().string(), response().url(), protectedDecoder()->encoding().name(), this);
}

String CachedCSSStyleSheet::responseMIMEType() const
{
    return extractMIMETypeFromMediaType(response().httpHeaderField(HTTPHeaderName::ContentType));
}

bool CachedCSSStyleSheet::mimeTypeAllowedByNosniff() const
{
    return parseContentTypeOptionsHeader(response().httpHeaderField(HTTPHeaderName::XContentTypeOptions)) != ContentTypeOptionsDisposition::Nosniff || equalLettersIgnoringASCIICase(responseMIMEType(), "text/css"_s);
}

bool CachedCSSStyleSheet::canUseSheet(MIMETypeCheckHint mimeTypeCheckHint, bool* hasValidMIMEType, bool* hasHTTPStatusOK) const
{
    if (errorOccurred())
        return false;

    // https://html.spec.whatwg.org/#fetching-and-processing-a-resource-from-a-link-element:processresponseconsumebody
    auto shouldAvoidUsingStyleSheetDueToUnsuccessfulRequest = [&] {
#if PLATFORM(COCOA)
        if (!linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::DoNotLoadStyleSheetIfHTTPStatusIsNotOK))
            return false;
#endif
        return response().url().protocolIsInHTTPFamily() && !response().isSuccessful();
    }();

    if (shouldAvoidUsingStyleSheetDueToUnsuccessfulRequest) {
        if (hasHTTPStatusOK)
            *hasHTTPStatusOK = false;
        return false;
    }

    if (!mimeTypeAllowedByNosniff()) {
        if (hasValidMIMEType)
            *hasValidMIMEType = false;
        return false;
    }

    if (mimeTypeCheckHint == MIMETypeCheckHint::Lax)
        return true;

    // This check exactly matches Firefox.  Note that we grab the Content-Type
    // header directly because we want to see what the value is BEFORE content
    // sniffing.  Firefox does this by setting a "type hint" on the channel.
    // This implementation should be observationally equivalent.
    //
    // This code defaults to allowing the stylesheet for non-HTTP protocols so
    // folks can use standards mode for local HTML documents.
    String mimeType = responseMIMEType();
    bool typeOK = mimeType.isEmpty() || equalLettersIgnoringASCIICase(mimeType, "text/css"_s) || equalLettersIgnoringASCIICase(mimeType, "application/x-unknown-content-type"_s) || !isValidContentType(mimeType);
    if (hasValidMIMEType)
        *hasValidMIMEType = typeOK;
    return typeOK;
}

void CachedCSSStyleSheet::destroyDecodedData()
{
    if (!m_parsedStyleSheetCache)
        return;

    Ref { *m_parsedStyleSheetCache }->removedFromMemoryCache();
    m_parsedStyleSheetCache = nullptr;

    setDecodedSize(0);
}

RefPtr<StyleSheetContents> CachedCSSStyleSheet::restoreParsedStyleSheet(const CSSParserContext& context, CachePolicy cachePolicy, FrameLoader& loader)
{
    RefPtr parsedStyleSheetCache = m_parsedStyleSheetCache;
    if (!parsedStyleSheetCache)
        return nullptr;
    if (!parsedStyleSheetCache->subresourcesAllowReuse(cachePolicy, loader)) {
        parsedStyleSheetCache->removedFromMemoryCache();
        m_parsedStyleSheetCache = nullptr;
        return nullptr;
    }

    ASSERT(parsedStyleSheetCache->isCacheable());
    ASSERT(parsedStyleSheetCache->isInMemoryCache());

    // Contexts must be identical so we know we would get the same exact result if we parsed again.
    if (parsedStyleSheetCache->parserContext() != context)
        return nullptr;

    didAccessDecodedData(MonotonicTime::now());

    return m_parsedStyleSheetCache;
}

void CachedCSSStyleSheet::saveParsedStyleSheet(Ref<StyleSheetContents>&& sheet)
{
    ASSERT(sheet->isCacheable());

    if (RefPtr parsedStyleSheetCache = m_parsedStyleSheetCache)
        parsedStyleSheetCache->removedFromMemoryCache();
    m_parsedStyleSheetCache = sheet.copyRef();
    sheet->addedToMemoryCache();

    setDecodedSize(sheet->estimatedSizeInBytes());
}

}
