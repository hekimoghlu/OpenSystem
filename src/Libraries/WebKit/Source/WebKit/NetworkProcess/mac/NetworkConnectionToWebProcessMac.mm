/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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
#import "NetworkConnectionToWebProcess.h"

#import "CoreIPCAuditToken.h"
#import <pal/spi/cocoa/LaunchServicesSPI.h>
#import <wtf/cocoa/VectorCocoa.h>

namespace WebKit {

#if PLATFORM(MAC)

void NetworkConnectionToWebProcess::updateActivePages(const String& overrideDisplayName, const Vector<String>& activePagesOrigins, CoreIPCAuditToken&& auditToken)
{
    // Setting and getting the display name of another process requires a private entitlement.
#if USE(APPLE_INTERNAL_SDK)
    auto asn = adoptCF(_LSCopyLSASNForAuditToken(kLSDefaultSessionID, auditToken.auditToken()));
    if (!overrideDisplayName)
        _LSSetApplicationInformationItem(kLSDefaultSessionID, asn.get(), CFSTR("LSActivePageUserVisibleOriginsKey"), (__bridge CFArrayRef)createNSArray(activePagesOrigins).get(), nullptr);
    else
        _LSSetApplicationInformationItem(kLSDefaultSessionID, asn.get(), _kLSDisplayNameKey, overrideDisplayName.createCFString().get(), nullptr);
#else
    UNUSED_PARAM(overrideDisplayName);
    UNUSED_PARAM(activePagesOrigins);
    UNUSED_PARAM(auditToken);
#endif
}

void NetworkConnectionToWebProcess::getProcessDisplayName(CoreIPCAuditToken&& auditToken, CompletionHandler<void(const String&)>&& completionHandler)
{
#if USE(APPLE_INTERNAL_SDK)
    auto asn = adoptCF(_LSCopyLSASNForAuditToken(kLSDefaultSessionID, auditToken.auditToken()));
    return completionHandler(adoptCF((CFStringRef)_LSCopyApplicationInformationItem(kLSDefaultSessionID, asn.get(), _kLSDisplayNameKey)).get());
#else
    completionHandler({ });
#endif
}

#endif // PLATFORM(MAC)

} // namespace WebKit
