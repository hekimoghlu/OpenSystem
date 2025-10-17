/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
#include "WKSecurityOriginRef.h"

#include "APISecurityOrigin.h"
#include "WKAPICast.h"
#include <WebCore/SecurityOriginData.h>

WKTypeID WKSecurityOriginGetTypeID()
{
    return WebKit::toAPI(API::SecurityOrigin::APIType);
}

WKSecurityOriginRef WKSecurityOriginCreateFromString(WKStringRef string)
{
    return WebKit::toAPI(&API::SecurityOrigin::create(WebCore::SecurityOrigin::createFromString(WebKit::toImpl(string)->string())).leakRef());
}

WKSecurityOriginRef WKSecurityOriginCreateFromDatabaseIdentifier(WKStringRef identifier)
{
    auto origin = WebCore::SecurityOriginData::fromDatabaseIdentifier(WebKit::toImpl(identifier)->string());
    if (!origin)
        return nullptr;
    return WebKit::toAPI(&API::SecurityOrigin::create(origin.value().securityOrigin()).leakRef());
}

WKSecurityOriginRef WKSecurityOriginCreate(WKStringRef protocol, WKStringRef host, int port)
{
    std::optional<uint16_t> validPort;
    if (port && port <= std::numeric_limits<uint16_t>::max())
        validPort = port;
    auto securityOrigin = API::SecurityOrigin::create(WebKit::toImpl(protocol)->string(), WebKit::toImpl(host)->string(), validPort);
    return WebKit::toAPI(&securityOrigin.leakRef());
}

WKStringRef WKSecurityOriginCopyDatabaseIdentifier(WKSecurityOriginRef securityOrigin)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(securityOrigin)->securityOrigin().databaseIdentifier());
}

WKStringRef WKSecurityOriginCopyToString(WKSecurityOriginRef securityOrigin)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(securityOrigin)->securityOrigin().toString());
}

WKStringRef WKSecurityOriginCopyProtocol(WKSecurityOriginRef securityOrigin)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(securityOrigin)->securityOrigin().protocol());
}

WKStringRef WKSecurityOriginCopyHost(WKSecurityOriginRef securityOrigin)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(securityOrigin)->securityOrigin().host());
}

unsigned short WKSecurityOriginGetPort(WKSecurityOriginRef securityOrigin)
{
    return WebKit::toImpl(securityOrigin)->securityOrigin().port().value_or(0);
}
