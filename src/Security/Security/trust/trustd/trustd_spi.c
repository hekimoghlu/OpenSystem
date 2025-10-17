/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
#ifdef LIBTRUSTD
#include <CoreFoundation/CoreFoundation.h>

#include "../utilities/SecFileLocations.h"
#include "../utilities/debugging.h"

#include "../sec/ipc/securityd_client.h"
#include "trust/trustd/SecPolicyServer.h"
#include "trust/trustd/SecTrustServer.h"
#include "trust/trustd/SecTrustSettingsServer.h"
#include "trust/trustd/SecTrustStoreServer.h"
#include "trust/trustd/SecOCSPCache.h"
#include "trust/trustd/OTATrustUtilities.h"
#include "trust/trustd/SecTrustLoggingServer.h"
#include "trust/trustd/SecRevocationDb.h"
#include "trust/trustd/SecPinningDb.h"
#include "trust/trustd/SecAnchorCache.h"
#include "trust/trustd/SecTrustExceptionResetCount.h"
#include "trust/trustd/trustdVariants.h"
#include "trustd_spi.h"

#if TARGET_OS_OSX
#include "trust/trustd/macOS/SecTrustOSXEntryPoints.h"
#endif

#endif // LIBTRUSTD

#ifdef LIBTRUSTD
struct trustd trustd_spi = {
    .sec_trust_store_for_domain             = SecTrustStoreForDomainName,
    .sec_trust_store_contains               = _SecTrustStoreContainsCertificate,
    .sec_trust_store_set_trust_settings     = _SecTrustStoreSetTrustSettings,
    .sec_trust_store_remove_certificate     = _SecTrustStoreRemoveCertificate,
    .sec_truststore_remove_all              = _SecTrustStoreRemoveAll,
    .sec_trust_evaluate                     = SecTrustServerEvaluate,
    .sec_ota_pki_trust_store_asset_version  = SecOTAPKICopyCurrentTrustStoreAssetVersion,
    .sec_ota_pki_trust_store_content_digest = SecOTAPKICopyCurrentTrustStoreContentDigest,
    .sec_ota_pki_trust_store_version        = SecOTAPKIGetCurrentTrustStoreVersion,
    .sec_ota_pki_asset_version              = SecOTATrustSupplementalsGetCurrentAssetVersion,
    .sec_ota_pki_copy_trusted_ct_logs       = SecOTAPKICopyCurrentTrustedCTLogs,
    .sec_ota_pki_copy_ct_log_for_keyid      = SecOTAPKICopyCTLogForKeyID,
    .sec_ota_pki_get_new_asset              = SecOTAPKISignalNewSupplementalsAsset,
    .sec_ota_secexperiment_get_new_asset    = SecOTASecExperimentGetNewAsset,
    .sec_ota_secexperiment_get_asset        = SecOTASecExperimentCopyAsset,
    .sec_trust_store_copy_all               = _SecTrustStoreCopyAll,
    .sec_trust_store_copy_usage_constraints = _SecTrustStoreCopyUsageConstraints,
    .sec_ocsp_cache_flush                   = SecOCSPCacheFlush,
    .sec_networking_analytics_report        = SecNetworkingAnalyticsReport,
    .sec_trust_store_set_ct_exceptions      = _SecTrustStoreSetCTExceptions,
    .sec_trust_store_copy_ct_exceptions     = _SecTrustStoreCopyCTExceptions,
    .sec_trust_get_exception_reset_count    = SecTrustServerGetExceptionResetCount,
    .sec_trust_increment_exception_reset_count = SecTrustServerIncrementExceptionResetCount,
    .sec_trust_store_set_ca_revocation_additions = _SecTrustStoreSetCARevocationAdditions,
    .sec_trust_store_copy_ca_revocation_additions = _SecTrustStoreCopyCARevocationAdditions,
    .sec_valid_update = SecRevocationDbUpdate,
    .sec_trust_store_set_transparent_connection_pins = _SecTrustStoreSetTransparentConnectionPins,
    .sec_trust_store_copy_transparent_connection_pins = _SecTrustStoreCopyTransparentConnectionPins,
    .sec_trust_settings_set_data            = SecTrustSettingsSetData,
    .sec_trust_settings_copy_data           = SecTrustSettingsCopyData,
    .sec_trust_reset_settings               = _SecTrustResetSettings,
    .sec_trust_store_migrate_plist          = SecTrustStoreMigratePropertyList,
};
#endif

void trustd_init(CFURLRef home_path) {
    if (home_path)
        SecSetCustomHomeURL(home_path);

    trustd_init_server();
}

void trustd_init_server(void) {
    gTrustd = &trustd_spi;
#ifdef LIBTRUSTD
    if (TrustdVariantAllowsFileWrite()) {
        CFErrorRef error = NULL;
        // Migrate files to data vault
        _SecTrustStoreMigrateConfigurations();
        _SecTrustStoreMigrateTrustSettings();
        SecTrustServerMigrateExceptionsResetCount();
        // Migrate trust store database content, updating schema if needed
        if (!_SecTrustStoreMigrateUserStore(&error)) {
            secerror("failed to migrate user trust store; new trust store will be empty: %@", error);
        }
        CFReleaseNull(error);
        // Migrate plist file content from data vault to trust store
        if (!_SecTrustStoreMigrateTrustSettingsPropertyList(&error)) {
            secerror("failed to migrate trust settings plist content: %@", error);
        }
        CFReleaseNull(error);
    }

    SecPolicyServerInitialize();    // set up callbacks for policy checks
    SecRevocationDbInitialize();    // set up revocation database if it doesn't already exist, or needs to be replaced
    SecPinningDbInitialize();       // set up the pinning database
    SecAnchorCacheInitialize();     // set up the anchor cache
#if TARGET_OS_OSX
    SecTrustLegacySourcesListenForKeychainEvents(); // set up the legacy keychain event listeners (for cache invalidation)
#endif
#endif  // LIBTRUSTD
}

void trustd_exit_clean(const char *reason) {
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 11ULL*NSEC_PER_SEC), dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        const char *default_reason = "Will exit trustd when all transactions are complete.";
        secnotice("OTATrust", "%s uptime: %llu, system: %llus",
                  (reason) ? reason : default_reason,
                  (unsigned long long)TimeSinceProcessLaunch(),
                  (unsigned long long)TimeSinceSystemStartup());
        xpc_transaction_exit_clean();
    });
}
