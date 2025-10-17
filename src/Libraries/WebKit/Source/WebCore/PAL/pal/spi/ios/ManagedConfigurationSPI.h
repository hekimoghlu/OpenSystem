/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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
#if PLATFORM(IOS_FAMILY) && !PLATFORM(MACCATALYST)

#if USE(APPLE_INTERNAL_SDK)

// FIXME: We conditionally enclose the ManagedConfiguration headers in an extern "C" linkage
// block to make them suitable for C++ use.
WTF_EXTERN_C_BEGIN

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 140000
#import <ManagedConfiguration/ManagedConfiguration.h>
@interface MCProfileConnection ()
- (NSArray<NSString *> *)crossSiteTrackingPreventionRelaxedDomains;
- (NSArray<NSString *> *)crossSiteTrackingPreventionRelaxedApps;
@end

#else
#import <ManagedConfiguration/MCFeatures.h>
#import <ManagedConfiguration/MCProfileConnection.h>
#endif

WTF_EXTERN_C_END

#else

WTF_EXTERN_C_BEGIN

extern NSString * const MCFeatureDefinitionLookupAllowed;

WTF_EXTERN_C_END

typedef enum MCRestrictedBoolType {
    MCRestrictedBoolExplicitYes = 1,
    MCRestrictedBoolExplicitNo = 2,
} MCRestrictedBoolType;

@interface MCProfileConnection : NSObject
@end

@class NSURL;

@interface MCProfileConnection ()
+ (MCProfileConnection *)sharedConnection;
- (MCRestrictedBoolType)effectiveBoolValueForSetting:(NSString *)feature;
- (BOOL)isURLManaged:(NSURL *)url;
- (NSArray<NSString *> *)crossSiteTrackingPreventionRelaxedDomains;
@end

#endif

#endif // PLATFORM(IOS_FAMILY) && !PLATFORM(MACCATALYST)
