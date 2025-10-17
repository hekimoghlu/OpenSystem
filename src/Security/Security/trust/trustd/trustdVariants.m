/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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
#import <Foundation/Foundation.h>
#import <os/variant_private.h>
#import <utilities/debugging.h>
#import "trustdVariants.h"
#include <MobileGestalt.h>

#if !TARGET_OS_BRIDGE
#import <MobileAsset/MAAsset.h>
#import <MobileAsset/MAAssetQuery.h>
#endif // !TARGET_OS_BRIDGE

bool TrustdVariantHasCertificatesBundle(void) {
#if TARGET_OS_BRIDGE
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        secnotice("trustd", "variant does not have certificates bundle");
    });
    return false;
#else
    return true;
#endif
}

bool TrustdVariantAllowsAnalytics(void) {
#if TARGET_OS_SIMULATOR || TARGET_OS_BRIDGE
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        secnotice("trustd", "variant does not allow analytics");
    });
    return false;
#else
    return TrustdVariantAllowsFileWrite();
#endif
}

bool TrustdVariantAllowsKeychain(void) {
#if TARGET_OS_BRIDGE
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        secnotice("trustd", "variant does not allow keychain");
    });
    return false;
#else
    return TrustdVariantAllowsFileWrite();
#endif
}

bool TrustdVariantAllowsFileWrite(void) {
    bool result = !os_variant_uses_ephemeral_storage("com.apple.security");
#if TARGET_OS_BRIDGE
    result = false;
#endif
    if (!result) {
        static dispatch_once_t onceToken;
        dispatch_once(&onceToken, ^{
            secnotice("trustd", "variant does not allow file writes");
        });
    }
    return result;
}

bool TrustdVariantAllowsNetwork(void) {
    // <rdar://32728029>
#if TARGET_OS_BRIDGE || TARGET_OS_WATCH
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        secnotice("trustd", "variant does not allow network");
    });
    return false;
#else
    return true;
#endif
}

bool TrustdVariantAllowsMobileAsset(void) {
    BOOL result = NO;
    if (TrustdVariantHasCertificatesBundle() && TrustdVariantAllowsFileWrite()) {
#if !TARGET_OS_BRIDGE
        /* MobileAsset.framework isn't mastered into the BaseSystem. Check that the MA classes are linked. */
        static dispatch_once_t onceToken;
        static BOOL classesAvailable = YES;
        dispatch_once(&onceToken, ^{
            if (![ASAssetQuery class] || ![ASAsset class] || ![MAAssetQuery class] || ![MAAsset class]) {
                secnotice("OTATrust", "Weak-linked MobileAsset framework missing.");
                classesAvailable = NO;
            }
        });
        if (classesAvailable) {
            result = YES;
        }
#endif
    }
    if (!result) {
        static dispatch_once_t onceToken;
        dispatch_once(&onceToken, ^{
            secnotice("trustd", "variant does not allow MobileAsset");
        });
    }
    return result;
}

bool TrustdVariantLowMemoryDevice(void) {
#if TARGET_OS_WATCH
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        secnotice("trustd", "low-memory variant");
    });
    return true;
#else
    return false;
#endif
}

bool TrustdVariantPrivateServerOS(void) {
    static BOOL result = NO;
    return result;
}
