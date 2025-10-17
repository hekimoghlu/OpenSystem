/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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

#if HAVE(SKADNETWORK_v4)
#if USE(APPLE_INTERNAL_SDK)

#if HAVE(ASD_INSTALL_WEB_ATTRIBUTION_SERVICE)
#import <AppStoreDaemon/ASDInstallWebAttributionService.h>
#endif

#import <AppStoreDaemon/ASDInstallWebAttributionParamsConfig.h>

#else // USE(APPLE_INTERNAL_SDK)

NS_ASSUME_NONNULL_BEGIN

@interface ASDInstallWebAttributionParamsConfig : NSObject <NSSecureCoding>
typedef NS_ENUM(NSInteger, ASDInstallWebAttributionContext) {
    AttributionTypeDefault = 0,
};
@property (nonatomic, strong) NSNumber *appAdamId;
@property (nonatomic, strong) NSString *adNetworkRegistrableDomain;
@property (nonatomic, strong) NSString *impressionId;
@property (nonatomic, strong) NSString *sourceWebRegistrableDomain;
@property (nonatomic, strong) NSString *version;
@property (nonatomic, assign) ASDInstallWebAttributionContext attributionContext;
@end

#if HAVE(ASD_INSTALL_WEB_ATTRIBUTION_SERVICE)
@interface ASDInstallWebAttributionService : NSObject
@property (class, readonly) ASDInstallWebAttributionService *sharedInstance;
- (void)addInstallWebAttributionParamsWithConfig:(ASDInstallWebAttributionParamsConfig *)config completionHandler:(nullable void (^)(NSError *error))completionHandler;
@end
#endif

NS_ASSUME_NONNULL_END

#endif // USE(APPLE_INTERNAL_SDK)

// FIXME: Move these to the !USE(APPLE_INTERNAL_SDK) section once rdar://137446922 is complete.
#if HAVE(AD_ATTRIBUTION_KIT_PRIVATE_BROWSING)
NS_ASSUME_NONNULL_BEGIN
@interface ASDInstallWebAttributionService (Staging_for_137446922)
- (void)removeInstallWebAttributionParamsFromPrivateBrowsingSessionID:(NSUUID *)sessionId completionHandler:(nullable void (^)(NSError *__nullable error))completionHandler;
@end
@interface ASDInstallWebAttributionParamsConfig (Staging_for_137446922)
@property (nullable, nonatomic, strong) NSUUID *privateBrowsingSessionId;
@end
NS_ASSUME_NONNULL_END
#endif // HAVE(AD_ATTRIBUTION_KIT_PRIVATE_BROWSING)

#endif // HAVE(SKADNETWORK_v4)
