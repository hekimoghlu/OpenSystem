/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 31, 2022.
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
/*
 * SecTrustOSXEntryPoints - Interface for unified SecTrust into OS X Security
 * Framework.
 */

#include "trust/trustd/macOS/SecTrustOSXEntryPoints.h"

#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#include <notify.h>

#include <Security/Security.h>
#include <Security/SecItemPriv.h>
#include <Security/SecTrustSettingsPriv.h>
#include <Security/SecItemInternal.h>

void SecTrustLegacySourcesListenForKeychainEvents(void) {
    /* Register for CertificateTrustNotification */
    int out_token = 0;
    notify_register_dispatch(kSecServerCertificateTrustNotification, &out_token,
                             dispatch_get_main_queue(),
                             ^(int token __unused) {
        // Purge keychain parent cache
        SecItemParentCachePurge();
        // Purge trust settings cert cache
        SecTrustSettingsPurgeUserAdminCertsCache();
        // Purge the trust settings cache
        SecTrustSettingsPurgeCache();
    });
}
