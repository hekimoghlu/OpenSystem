/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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
#include "WKProtectionSpace.h"

#include "WebProtectionSpace.h"
#include "WKAPICast.h"

using namespace WebKit;

WKTypeID WKProtectionSpaceGetTypeID()
{
    return toAPI(WebProtectionSpace::APIType);
}

WKStringRef WKProtectionSpaceCopyHost(WKProtectionSpaceRef protectionSpaceRef)
{
    return toCopiedAPI(toImpl(protectionSpaceRef)->host());
}

int WKProtectionSpaceGetPort(WKProtectionSpaceRef protectionSpaceRef)
{
    return toImpl(protectionSpaceRef)->port();
}

WKStringRef WKProtectionSpaceCopyRealm(WKProtectionSpaceRef protectionSpaceRef)
{
    return toCopiedAPI(toImpl(protectionSpaceRef)->realm());
}

bool WKProtectionSpaceGetIsProxy(WKProtectionSpaceRef protectionSpaceRef)
{
    return toImpl(protectionSpaceRef)->isProxy();
}

WKProtectionSpaceServerType WKProtectionSpaceGetServerType(WKProtectionSpaceRef protectionSpaceRef)
{
    return toAPI(toImpl(protectionSpaceRef)->serverType());
}

bool WKProtectionSpaceGetReceivesCredentialSecurely(WKProtectionSpaceRef protectionSpaceRef)
{
    return toImpl(protectionSpaceRef)->receivesCredentialSecurely();
}

WKProtectionSpaceAuthenticationScheme WKProtectionSpaceGetAuthenticationScheme(WKProtectionSpaceRef protectionSpaceRef)
{
    return toAPI(toImpl(protectionSpaceRef)->authenticationScheme());
}
