/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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
#include "ResourceError.h"

#if USE(SOUP)

#include "LocalizedStrings.h"
#include "URLSoup.h"
#include <libsoup/soup.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/text/CString.h>

namespace WebCore {

#if USE(SOUP2)
#define SOUP_HTTP_ERROR_DOMAIN SOUP_HTTP_ERROR
#else
#define SOUP_HTTP_ERROR_DOMAIN SOUP_SESSION_ERROR
#endif

ResourceError::ResourceError(const String& domain, int errorCode, const URL& failingURL, const String& localizedDescription, Type type, IsSanitized isSanitized)
    : ResourceErrorBase(domain, errorCode, failingURL, localizedDescription, type, isSanitized)
{
}

ResourceError ResourceError::fromIPCData(std::optional<IPCData>&& ipcData)
{
    if (!ipcData)
        return { };

    ResourceError error {
        ipcData->domain,
        ipcData->errorCode,
        ipcData->failingURL,
        ipcData->localizedDescription,
        ipcData->type,
        ipcData->isSanitized
    };
    error.setCertificate(ipcData->certificateInfo.certificate().get());
    error.setTLSErrors(ipcData->certificateInfo.tlsErrors());
    return error;
}

auto ResourceError::ipcData() const -> std::optional<IPCData>
{
    if (isNull())
        return std::nullopt;

    return IPCData {
        type(),
        domain(),
        errorCode(),
        failingURL(),
        localizedDescription(),
        m_isSanitized,
        CertificateInfo { *this }
    };
}

ResourceError ResourceError::transportError(const URL& failingURL, int statusCode, const String& reasonPhrase)
{
    return ResourceError(String::fromLatin1(g_quark_to_string(SOUP_HTTP_ERROR_DOMAIN)), statusCode, failingURL, reasonPhrase);
}

ResourceError ResourceError::httpError(SoupMessage* message, GError* error)
{
    ASSERT(message);
#if USE(SOUP2)
    if (SOUP_STATUS_IS_TRANSPORT_ERROR(message->status_code))
        return transportError(soupURIToURL(soup_message_get_uri(message)), message->status_code, String::fromUTF8(message->reason_phrase));
#endif
    return genericGError(soupURIToURL(soup_message_get_uri(message)), error);
}

ResourceError ResourceError::authenticationError(SoupMessage* message)
{
    ASSERT(message);
#if USE(SOUP2)
    return ResourceError(String::fromLatin1(g_quark_to_string(SOUP_HTTP_ERROR_DOMAIN)), message->status_code,
        soupURIToURL(soup_message_get_uri(message)), String::fromUTF8(message->reason_phrase));
#else
    return ResourceError(String::fromLatin1(g_quark_to_string(SOUP_SESSION_ERROR)), soup_message_get_status(message),
        soup_message_get_uri(message), String::fromUTF8(soup_message_get_reason_phrase(message)));
#endif
}

ResourceError ResourceError::genericGError(const URL& failingURL, GError* error)
{
    return ResourceError(String::fromLatin1(g_quark_to_string(error->domain)), error->code, failingURL, String::fromUTF8(error->message));
}

ResourceError ResourceError::tlsError(const URL& failingURL, unsigned tlsErrors, GTlsCertificate* certificate)
{
    ResourceError resourceError(String::fromLatin1(g_quark_to_string(G_TLS_ERROR)), G_TLS_ERROR_BAD_CERTIFICATE, failingURL, unacceptableTLSCertificate());
    resourceError.setTLSErrors(tlsErrors);
    resourceError.setCertificate(certificate);
    return resourceError;
}

ResourceError ResourceError::timeoutError(const URL& failingURL)
{
    // FIXME: This should probably either be integrated into ErrorsGtk.h or the
    // networking errors from that file should be moved here.

    // Use the same value as in NSURLError.h
    static constexpr int timeoutError = -1001;
    static constexpr auto errorDomain = "WebKitNetworkError"_s;
    return ResourceError(errorDomain, timeoutError, failingURL, "Request timed out"_s, ResourceError::Type::Timeout);
}

void ResourceError::doPlatformIsolatedCopy(const ResourceError& other)
{
    m_certificate = other.m_certificate;
    m_tlsErrors = other.m_tlsErrors;
}

bool ResourceError::platformCompare(const ResourceError& a, const ResourceError& b)
{
    return a.tlsErrors() == b.tlsErrors();
}

} // namespace WebCore

#endif
