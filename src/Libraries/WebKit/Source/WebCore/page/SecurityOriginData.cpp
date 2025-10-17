/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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
#include "SecurityOriginData.h"

#include "BlobURL.h"
#include "Document.h"
#include "LegacySchemeRegistry.h"
#include "LocalFrame.h"
#include "SecurityOrigin.h"
#include <wtf/FileSystem.h>
#include <wtf/text/CString.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

String SecurityOriginData::toString() const
{
    auto protocol = this->protocol();
    if (protocol == "file"_s)
        return "file://"_s;

    auto host = this->host();
    if (protocol.isEmpty() && host.isEmpty())
        return { };

    auto port = this->port();
    if (!port)
        return makeString(protocol, "://"_s, host);
    return makeString(protocol, "://"_s, host, ':', static_cast<uint32_t>(*port));
}

URL SecurityOriginData::toURL() const
{
    return URL { toString() };
}

SecurityOriginData SecurityOriginData::fromFrame(LocalFrame* frame)
{
    if (!frame)
        return SecurityOriginData { };
    
    auto* document = frame->document();
    if (!document)
        return SecurityOriginData { };

    return document->securityOrigin().data();
}

SecurityOriginData SecurityOriginData::fromURL(const URL& url)
{
    if (shouldTreatAsOpaqueOrigin(url))
        return createOpaque();
    return fromURLWithoutStrictOpaqueness(url);
}

SecurityOriginData SecurityOriginData::fromURLWithoutStrictOpaqueness(const URL& url)
{
    if (url.isNull())
        return SecurityOriginData { };
    if (url.protocol().isEmpty() && url.host().isEmpty() && !url.port())
        return createOpaque();
    return SecurityOriginData {
        url.protocol().isNull() ? emptyString() : url.protocol().convertToASCIILowercase()
        , url.host().isNull() ? emptyString() : url.host().convertToASCIILowercase()
        , url.port()
    };
}

Ref<SecurityOrigin> SecurityOriginData::securityOrigin() const
{
    return SecurityOrigin::create(isolatedCopy());
}

static const char separatorCharacter = '_';

String SecurityOriginData::databaseIdentifier() const
{
    // Historically, we've used the following (somewhat nonsensical) string
    // for the databaseIdentifier of local files. We used to compute this
    // string because of a bug in how we handled the scheme for file URLs.
    // Now that we've fixed that bug, we produce this string for compatibility
    // with existing persistent state.
    auto protocol = this->protocol();
    if (equalLettersIgnoringASCIICase(protocol, "file"_s))
        return "file__0"_s;

    return makeString(protocol, separatorCharacter, FileSystem::encodeForFileName(host()), separatorCharacter, port().value_or(0));
}

String SecurityOriginData::optionalDatabaseIdentifier() const
{
    auto url = toURL();
    if (!url.isValid())
        return { };

    return databaseIdentifier();
}

std::optional<SecurityOriginData> SecurityOriginData::fromDatabaseIdentifier(StringView databaseIdentifier)
{
    // Make sure there's a first separator
    size_t separator1 = databaseIdentifier.find(separatorCharacter);
    if (separator1 == notFound)
        return std::nullopt;
    
    // Make sure there's a second separator
    size_t separator2 = databaseIdentifier.reverseFind(separatorCharacter);
    if (separator2 == notFound)
        return std::nullopt;
    
    // Ensure there were at least 2 separator characters. Some hostnames on intranets have
    // underscores in them, so we'll assume that any additional underscores are part of the host.
    if (separator1 == separator2)
        return std::nullopt;
    
    // Make sure the port section is a valid port number or doesn't exist.
    auto portLength = databaseIdentifier.length() - separator2 - 1;
    auto port = parseIntegerAllowingTrailingJunk<uint16_t>(databaseIdentifier.right(portLength));

    // Nothing after the colon is fine. Failure to parse after the colon is not.
    if (!port && portLength)
        return std::nullopt;

    // Treat port 0 like there is was no port specified.
    if (port && !*port)
        port = std::nullopt;

    auto protocol = databaseIdentifier.left(separator1);
    auto host = databaseIdentifier.substring(separator1 + 1, separator2 - separator1 - 1);
    return SecurityOriginData { protocol.toString(), host.toString(), port };
}

SecurityOriginData SecurityOriginData::isolatedCopy() const &
{
    return SecurityOriginData { crossThreadCopy(m_data) };
}

SecurityOriginData SecurityOriginData::isolatedCopy() &&
{
    return SecurityOriginData { crossThreadCopy(WTFMove(m_data)) };
}

bool operator==(const SecurityOriginData& a, const SecurityOriginData& b)
{
    if (&a == &b)
        return true;

    return a.data() == b.data();
}

static bool schemeRequiresHost(const URL& url)
{
    // We expect URLs with these schemes to have authority components. If the
    // URL lacks an authority component, we get concerned and mark the origin
    // as opaque.
    return url.protocolIsInHTTPFamily() || url.protocolIs("ftp"_s);
}

bool SecurityOriginData::shouldTreatAsOpaqueOrigin(const URL& url)
{
    if (!url.isValid())
        return true;

    auto originURL = url.protocolIsBlob() ? BlobURL::getOriginURL(url) : url;
    if (!originURL.isValid())
        return true;

    // For edge case URLs that were probably misparsed, make sure that the origin is opaque.
    // This is an additional safety net against bugs in URL parsing, and for network back-ends that parse URLs differently,
    // and could misinterpret another component for hostname.
    if (schemeRequiresHost(originURL) && originURL.host().isEmpty())
        return true;

    if (LegacySchemeRegistry::shouldTreatURLSchemeAsNoAccess(originURL.protocol()))
        return true;

    // https://url.spec.whatwg.org/#origin with some additions
    if (url.hasSpecialScheme()
#if PLATFORM(COCOA)
        || !linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::NullOriginForNonSpecialSchemedURLs)
        || url.protocolIs("applewebdata"_s)
        || url.protocolIs("x-apple-ql-id"_s)
        || url.protocolIs("x-apple-ql-id2"_s)
        || url.protocolIs("x-apple-ql-magic"_s)
#endif
#if PLATFORM(GTK) || PLATFORM(WPE)
        || url.protocolIs("resource"_s)
#endif
#if ENABLE(PDFJS)
        || url.protocolIs("webkit-pdfjs-viewer"_s)
#endif
        || url.protocolIsBlob())
        return false;

    // FIXME: we ought to assert we're in WebKitLegacy or a web content process as per 263652@main,
    // except that assert gets hit on certain tests.
    return !LegacySchemeRegistry::schemeIsHandledBySchemeHandler(url.protocol());
}

} // namespace WebCore
