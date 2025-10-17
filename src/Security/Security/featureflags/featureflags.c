/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
// For functions related to feature flags used in clients/frameworks and servers/daemons

#include "featureflags.h"

#include <stdatomic.h>
#include <dispatch/dispatch.h>
#include <os/feature_private.h>
#include <os/variant_private.h>
#include <security_utilities/debugging.h>


// feature flag for supporting system keychain on non-edu-mode iOS

typedef enum {
    SystemKeychainAlways_DEFAULT,
    SystemKeychainAlways_OVERRIDE_TRUE,
    SystemKeychainAlways_OVERRIDE_FALSE,
} SystemKeychainAlwaysSupported;

static SystemKeychainAlwaysSupported gSystemKeychainAlwaysSupported = SystemKeychainAlways_DEFAULT;

bool _SecSystemKeychainAlwaysIsEnabled(void)
{
    if (gSystemKeychainAlwaysSupported != SystemKeychainAlways_DEFAULT) {
        return gSystemKeychainAlwaysSupported == SystemKeychainAlways_OVERRIDE_TRUE;
    }

    static bool ffSystemKeychainAlwaysSupported = false;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
#if TARGET_OS_OSX
        ffSystemKeychainAlwaysSupported = true;
        secnotice("keychain", "Enabling System Keychain Always due to platform");
#else
        ffSystemKeychainAlwaysSupported = os_feature_enabled(Security, SecSystemKeychainAlwaysSupported);
        secnotice("keychain", "System Keychain Always Supported set via feature flag to %s", ffSystemKeychainAlwaysSupported ? "enabled" : "disabled");
#endif
    });

    return ffSystemKeychainAlwaysSupported;
}

void _SecSystemKeychainAlwaysOverride(bool value)
{
    gSystemKeychainAlwaysSupported = value ? SystemKeychainAlways_OVERRIDE_TRUE : SystemKeychainAlways_OVERRIDE_FALSE;
    secnotice("keychain", "System Keychain Always Supported overridden to %s", value ? "enabled" : "disabled");
}

void _SecSystemKeychainAlwaysClearOverride(void)
{
    gSystemKeychainAlwaysSupported = SystemKeychainAlways_DEFAULT;
    secnotice("keychain", "System Keychain Always Supported override removed");
}

static void _SecTrustShowFeatureStatus(const char* feature, bool status) {
    secnotice("trustd", "%s is %s (via feature flags)",
              feature, status ? "enabled" : "disabled");
}

bool _SecTrustQWACValidationEnabled(void)
{
    /* NOTE: This feature flags are referenced by string in unit tests.
     * If you're here cleaning up, please remove it from the tests as well. */
    static bool QWACValidationEnabled = false;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        QWACValidationEnabled = os_feature_enabled(Security, QWACValidation);
        _SecTrustShowFeatureStatus("QWACValidation", QWACValidationEnabled);
    });
    return QWACValidationEnabled;
}

bool _SecTrustStoreRootConstraintsEnabled(void)
{
    /* NOTE: This feature flags are referenced by string in unit tests.
     * If you're here cleaning up, please remove it from the tests as well. */
    static bool RootConstraintsEnabled = false;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        RootConstraintsEnabled = os_feature_enabled(Security, RootConstraints);
        _SecTrustShowFeatureStatus("RootConstraints", RootConstraintsEnabled);
    });
    return RootConstraintsEnabled;
}
