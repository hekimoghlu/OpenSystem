/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#import "CoreTelephonyUtilities.h"

#if HAVE(CORE_TELEPHONY)

#import "DefaultWebBrowserChecks.h"
#import "Logging.h"
#import <WebCore/RegistrableDomain.h>
#import <wtf/RetainPtr.h>
#import <wtf/cf/TypeCastsCF.h>
#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/spi/cocoa/SecuritySPI.h>

#import <pal/cocoa/CoreTelephonySoftLink.h>

namespace WebKit {

#if HAVE(ESIM_AUTOFILL_SYSTEM_SUPPORT)

bool shouldAllowAutoFillForCellularIdentifiers(const URL& topURL)
{
    static bool hasLogged = false;
    if (isFullWebBrowserOrRunningTest()) {
        if (!std::exchange(hasLogged, true))
            RELEASE_LOG(Telephony, "Skipped cellular AutoFill status check (app is a web browser)");
        return false;
    }

    static bool hasPublicCellularPlanEntitlement = [] {
        auto task = adoptCF(SecTaskCreateFromSelf(kCFAllocatorDefault));
        if (!task)
            return false;

        auto entitlementValue = adoptCF(SecTaskCopyValueForEntitlement(task.get(), CFSTR("com.apple.CommCenter.fine-grained"), nullptr));
        auto entitlementValueAsArray = (__bridge NSArray *)dynamic_cf_cast<CFArrayRef>(entitlementValue.get());
        for (id value in entitlementValueAsArray) {
            if (auto string = dynamic_objc_cast<NSString>(value); [string isEqualToString:@"public-cellular-plan"])
                return true;
        }
        return false;
    }();

    if (!hasPublicCellularPlanEntitlement) {
        if (!std::exchange(hasLogged, true))
            RELEASE_LOG(Telephony, "Skipped cellular AutoFill status check (app does not have cellular plan entitlement)");
        return false;
    }

    auto host = topURL.host().toString();
    if (host.isEmpty()) {
        if (!std::exchange(hasLogged, true))
            RELEASE_LOG(Telephony, "Skipped cellular AutoFill status check (no registrable domain)");
        return false;
    }

#if HAVE(DELAY_INIT_LINKING)
    static NeverDestroyed cachedClient = adoptNS([[CoreTelephonyClient alloc] initWithQueue:dispatch_get_main_queue()]);
#else
    static NeverDestroyed cachedClient = adoptNS([PAL::allocCoreTelephonyClientInstance() initWithQueue:dispatch_get_main_queue()]);
#endif
    auto client = cachedClient->get();

    static NeverDestroyed<String> lastQueriedHost;
    static bool lastQueriedHostResult = false;
    if (lastQueriedHost.get() == host)
        return lastQueriedHostResult;

    NSError *error = nil;
    BOOL result = [client isAutofilleSIMIdAllowedForDomain:host error:&error];
    if (error && !std::exchange(hasLogged, true)) {
        RELEASE_LOG_ERROR(Telephony, "Failed to query cellular AutoFill status: %{public}@", error);
        return false;
    }

    if (!std::exchange(hasLogged, true))
        RELEASE_LOG(Telephony, "Is cellular AutoFill allowed for current host? %{public}@", result ? @"YES" : @"NO");

    lastQueriedHost.get() = WTFMove(host);
    lastQueriedHostResult = !!result;
    return lastQueriedHostResult;
}

#endif // HAVE(ESIM_AUTOFILL_SYSTEM_SUPPORT)

} // namespace WebKit

#endif // HAVE(CORE_TELEPHONY)
