/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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
#import "ResourceResponse.h"

#if PLATFORM(COCOA)

#import "HTTPParsers.h"
#import "WebCoreURLResponse.h"
#import <Foundation/Foundation.h>
#import <limits>
#import <pal/spi/cf/CFNetworkSPI.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/StdLibExtras.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/cf/TypeCastsCF.h>
#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/text/StringView.h>

WTF_DECLARE_CF_TYPE_TRAIT(SecTrust);

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ResourceResponse);

void ResourceResponse::initNSURLResponse() const
{
    if (!m_httpStatusCode || !m_url.protocolIsInHTTPFamily()) {
        // Work around a mistake in the NSURLResponse class - <rdar://problem/6875219>.
        // The init function takes an NSInteger, even though the accessor returns a long long.
        // For values that won't fit in an NSInteger, pass -1 instead.
        NSInteger expectedContentLength;
        if (m_expectedContentLength < 0 || m_expectedContentLength > std::numeric_limits<NSInteger>::max())
            expectedContentLength = -1;
        else
            expectedContentLength = static_cast<NSInteger>(m_expectedContentLength);

        NSString* encodingNSString = nsStringNilIfEmpty(m_textEncodingName);
        m_nsResponse = adoptNS([[NSURLResponse alloc] initWithURL:m_url MIMEType:m_mimeType expectedContentLength:expectedContentLength textEncodingName:encodingNSString]);
        return;
    }

    // FIXME: We lose the status text and the HTTP version here.
    NSMutableDictionary* headerDictionary = [NSMutableDictionary dictionary];
    for (auto& header : m_httpHeaderFields)
        [headerDictionary setObject:(NSString *)header.value forKey:(NSString *)header.key];

    m_nsResponse = adoptNS([[NSHTTPURLResponse alloc] initWithURL:m_url statusCode:m_httpStatusCode HTTPVersion:(NSString*)kCFHTTPVersion1_1 headerFields:headerDictionary]);

    // Mime type sniffing doesn't work with a synthesized response.
    [m_nsResponse _setMIMEType:(NSString *)m_mimeType];
}

void ResourceResponse::disableLazyInitialization()
{
    lazyInit(AllFields);
}

CertificateInfo ResourceResponse::platformCertificateInfo(std::span<const std::byte> auditToken) const
{
    CFURLResponseRef cfResponse = [m_nsResponse _CFURLResponse];
    if (!cfResponse)
        return { };

    CFDictionaryRef context = _CFURLResponseGetSSLCertificateContext(cfResponse);
    if (!context)
        return { };

    auto trustValue = CFDictionaryGetValue(context, kCFStreamPropertySSLPeerTrust);
    if (!trustValue)
        return { };
    auto trust = checked_cf_cast<SecTrustRef>(trustValue);

    if (trust && auditToken.size()) {
        auto data = adoptCF(CFDataCreate(nullptr, byteCast<uint8_t>(auditToken.data()), auditToken.size()));
        SecTrustSetClientAuditToken(trust, data.get());
    }

    SecTrustResultType trustResultType;
    OSStatus result = SecTrustGetTrustResult(trust, &trustResultType);
    if (result != errSecSuccess)
        return { };

    if (trustResultType == kSecTrustResultInvalid) {
        if (!SecTrustEvaluateWithError(trust, nullptr))
            return { };
    }

    return CertificateInfo(trust);
}

NSURLResponse *ResourceResponse::nsURLResponse() const
{
    if (!m_nsResponse && !m_isNull)
        initNSURLResponse();
    return m_nsResponse.get();
}

static void addToHTTPHeaderMap(const void* key, const void* value, void* context)
{
    HTTPHeaderMap* httpHeaderMap = (HTTPHeaderMap*)context;
    httpHeaderMap->set((CFStringRef)key, (CFStringRef)value);
}

static inline AtomString stripLeadingAndTrailingDoubleQuote(const String& value)
{
    unsigned length = value.length();
    if (length < 2 || value[0u] != '"' || value[length - 1] != '"')
        return AtomString { value };

    return StringView(value).substring(1, length - 2).toAtomString();
}

static inline HTTPHeaderMap initializeHTTPHeaders(CFHTTPMessageRef messageRef)
{
    // Avoid calling [NSURLResponse allHeaderFields] to minimize copying (<rdar://problem/26778863>).
    auto headers = adoptCF(CFHTTPMessageCopyAllHeaderFields(messageRef));

    HTTPHeaderMap headersMap;
    CFDictionaryApplyFunction(headers.get(), addToHTTPHeaderMap, &headersMap);
    return headersMap;
}

static inline AtomString extractHTTPStatusText(CFHTTPMessageRef messageRef)
{
    if (auto httpStatusLine = adoptCF(CFHTTPMessageCopyResponseStatusLine(messageRef)))
        return extractReasonPhraseFromHTTPStatusLine(httpStatusLine.get());

    static MainThreadNeverDestroyed<const AtomString> defaultStatusText("OK"_s);
    return defaultStatusText;
}

void ResourceResponse::platformLazyInit(InitLevel initLevel)
{
    ASSERT(initLevel >= CommonFieldsOnly);

    if (m_initLevel >= initLevel)
        return;

    if (m_isNull || !m_nsResponse)
        return;
    
    @autoreleasepool {

        RetainPtr urlResponse = dynamic_objc_cast<NSHTTPURLResponse>(m_nsResponse.get());
        RetainPtr messageRef = urlResponse ? CFURLResponseGetHTTPResponse([urlResponse _CFURLResponse]) : nullptr;

        if (m_initLevel < CommonFieldsOnly) {
            m_url = [m_nsResponse URL];
            m_mimeType = [m_nsResponse MIMEType];
            m_expectedContentLength = [m_nsResponse expectedContentLength];
            // Stripping double quotes as a workaround for <rdar://problem/8757088>, can be removed once that is fixed.
            m_textEncodingName = stripLeadingAndTrailingDoubleQuote([m_nsResponse textEncodingName]);
            m_httpStatusCode = messageRef ? CFHTTPMessageGetResponseStatusCode(messageRef.get()) : 0;
            if (messageRef)
                m_httpHeaderFields = initializeHTTPHeaders(messageRef.get());
        }
        if (messageRef && initLevel == AllFields) {
            m_httpStatusText = extractHTTPStatusText(messageRef.get());
            m_httpVersion = AtomString { String(adoptCF(CFHTTPMessageCopyVersion(messageRef.get())).get()).convertToASCIIUppercase() };
        }
    }

    m_initLevel = initLevel;
}

String ResourceResponse::platformSuggestedFilename() const
{
    return [nsURLResponse() suggestedFilename];
}

bool ResourceResponse::platformCompare(const ResourceResponse& a, const ResourceResponse& b)
{
    return a.nsURLResponse() == b.nsURLResponse();
}

} // namespace WebCore

#endif // PLATFORM(COCOA)
