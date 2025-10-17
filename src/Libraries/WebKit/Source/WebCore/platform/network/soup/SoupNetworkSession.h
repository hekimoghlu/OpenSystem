/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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
#pragma once

#include "SoupNetworkProxySettings.h"
#include <gio/gio.h>
#include <glib-object.h>
#include <pal/SessionID.h>
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

typedef struct _SoupCache SoupCache;
typedef struct _SoupCookieJar SoupCookieJar;
typedef struct _SoupMessage SoupMessage;
typedef struct _SoupSession SoupSession;

namespace WebCore {

class CertificateInfo;
class HostTLSCertificateSet;
class ResourceError;

class SoupNetworkSession {
    WTF_MAKE_TZONE_ALLOCATED(SoupNetworkSession);
    WTF_MAKE_NONCOPYABLE(SoupNetworkSession);
public:
    explicit SoupNetworkSession(PAL::SessionID);
    ~SoupNetworkSession();

    SoupSession* soupSession() const { return m_soupSession.get(); }

    void setCookieJar(SoupCookieJar*);
    SoupCookieJar* cookieJar() const;

    void setHSTSPersistentStorage(const String& hstsStorageDirectory);

    static void clearOldSoupCache(const String& cacheDirectory);

    void setProxySettings(const SoupNetworkProxySettings&);

    static void setInitialAcceptLanguages(const CString&);
    void setAcceptLanguages(const CString&);

    WEBCORE_EXPORT void setIgnoreTLSErrors(bool);
    std::optional<ResourceError> checkTLSErrors(const URL&, GTlsCertificate*, GTlsCertificateFlags);
    void allowSpecificHTTPSCertificateForHost(const CertificateInfo&, const String& host);

    void getHostNamesWithHSTSCache(HashSet<String>&);
    void deleteHSTSCacheForHostNames(const Vector<String>&);
    void clearHSTSCache(WallTime);

private:
    void setupLogger();

    GRefPtr<SoupSession> m_soupSession;
    PAL::SessionID m_sessionID;
    bool m_ignoreTLSErrors { false };
    SoupNetworkProxySettings m_proxySettings;
    HashMap<String, HostTLSCertificateSet, ASCIICaseInsensitiveHash> m_allowedCertificates;
};

} // namespace WebCore
