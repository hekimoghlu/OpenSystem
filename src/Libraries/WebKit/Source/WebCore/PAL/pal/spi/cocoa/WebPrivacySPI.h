/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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

#if ENABLE(ADVANCED_PRIVACY_PROTECTIONS)

#if HAVE(WEB_PRIVACY_FRAMEWORK)
#import <WebPrivacy/WebPrivacy.h>
#else

#import <Foundation/Foundation.h>

typedef NS_ENUM(NSInteger, WPResourceType) {
    WPResourceTypeTrackerBlockList = 1,
    WPResourceTypeLinkFilteringData,
    WPResourceTypeTrackerDomains,
    WPResourceTypeTrackerNetworkAddresses,
    WPResourceTypeAllowedLinkFilteringData,
};

typedef NS_ENUM(NSInteger, WPNetworkAddressVersion) {
    WPNetworkAddressVersion4 = 4,
    WPNetworkAddressVersion6 = 6
};

@interface WPNetworkAddressRange : NSObject
@property (nonatomic, readonly) WPNetworkAddressVersion version;
@property (nonatomic, readonly) const struct sockaddr* address;
@property (nonatomic, readonly) NSUInteger netMaskLength;
@property (nonatomic, readonly) NSString *owner;
@property (nonatomic, readonly) NSString *host;
@end

@interface WPResourceRequestOptions : NSObject
@property (nonatomic) BOOL afterUpdates;
@end

@interface WPLinkFilteringRule : NSObject
@property (nonatomic, readonly) NSString *queryParameter;
@property (nonatomic, readonly) NSString *domain;
@property (nonatomic, readonly) NSString *path;
@end

@interface WPLinkFilteringData : NSObject
@property (nonatomic, readonly) NSArray<WPLinkFilteringRule *> *rules;
@end

@interface WPTrackingDomain : NSObject
@property (nonatomic, readonly) NSString *host;
@property (nonatomic, readonly) NSString *owner;
@property (nonatomic, readonly) BOOL canBlock;
@end

typedef void (^WPNetworkAddressesCompletionHandler)(NSArray<WPNetworkAddressRange *> *, NSError *);
typedef void (^WPLinkFilteringDataCompletionHandler)(WPLinkFilteringData *, NSError *);
typedef void (^WPTrackingDomainsCompletionHandler)(NSArray<WPTrackingDomain *> *, NSError *);

@interface WPResources : NSObject

+ (instancetype)sharedInstance;

- (void)requestTrackerNetworkAddresses:(WPResourceRequestOptions *)options completionHandler:(WPNetworkAddressesCompletionHandler)completion;
- (void)requestLinkFilteringData:(WPResourceRequestOptions *)options completionHandler:(WPLinkFilteringDataCompletionHandler)completion;
- (void)requestAllowedLinkFilteringData:(WPResourceRequestOptions *)options completionHandler:(WPLinkFilteringDataCompletionHandler)completion;
- (void)requestTrackerDomainNamesData:(WPResourceRequestOptions *)options completionHandler:(WPTrackingDomainsCompletionHandler)completion;

@end

#endif // !HAVE(WEB_PRIVACY_FRAMEWORK)

#if !defined(HAS_WEB_PRIVACY_STORAGE_ACCESS_PROMPT_QUIRK_CLASS)
constexpr NSInteger WPResourceTypeStorageAccessPromptQuirksData = 7;

@interface WPStorageAccessPromptQuirk : NSObject
@property (nonatomic, readonly) NSString *name;
@property (nonatomic, readonly) NSDictionary<NSString *, NSArray<NSString *> *> *domainPairings;
@property (nonatomic, readonly) NSDictionary<NSString *, NSArray<NSString *> *> *quirkDomains;
@property (nonatomic, readonly) NSArray<NSString *> *triggerPages;
@end

@interface WPStorageAccessPromptQuirksData : NSObject
@property (nonatomic, readonly) NSArray<WPStorageAccessPromptQuirk *> *quirks;
@end

typedef void (^WPStorageAccessPromptQuirksDataCompletionHandler)(WPStorageAccessPromptQuirksData *, NSError *);

@interface WPResources (Staging_119342418_PromptQuirks)
- (void)requestStorageAccessPromptQuirksData:(WPResourceRequestOptions *)options completionHandler:(WPStorageAccessPromptQuirksDataCompletionHandler)completion;
@end
#endif

#if !defined(HAS_WEB_PRIVACY_STORAGE_ACCESS_USER_AGENT_STRING_CLASS)
constexpr NSInteger WPResourceTypeStorageAccessUserAgentStringQuirksData = 6;
@interface WPStorageAccessUserAgentStringQuirk : NSObject
@property (nonatomic, readonly) NSString *domain;
@property (nonatomic, readonly) NSString *userAgentString;
@end

@interface WPStorageAccessUserAgentStringQuirksData : NSObject
@property (nonatomic, readonly) NSArray<WPStorageAccessUserAgentStringQuirk *> *quirks;
@end

typedef void (^WPStorageAccessUserAgentStringQuirksDataCompletionHandler)(WPStorageAccessUserAgentStringQuirksData *, NSError *);

@interface WPResources (Staging_119342418_UAQuirks)
- (void)requestStorageAccessUserAgentStringQuirksData:(WPResourceRequestOptions *)options completionHandler:(WPStorageAccessUserAgentStringQuirksDataCompletionHandler)completion;
@end
#endif

#if !defined(HAS_WEB_PRIVACY_LINK_FILTERING_RULE_PATH) && HAVE(WEB_PRIVACY_FRAMEWORK)
@interface WPLinkFilteringRule (Staging_119590894)
@property (nonatomic, readonly) NSString *path;
@end
#endif

#if !defined(HAS_WEB_PRIVACY_RESTRICTED_OPENER_DOMAIN_CLASS)
constexpr NSInteger WPResourceTypeRestrictedOpenerDomains = 8;

typedef NS_ENUM(NSInteger, WPRestrictedOpenerType) {
    WPRestrictedOpenerTypeNoOpener = 1,
    WPRestrictedOpenerTypePostMessageAndClose,
};

@interface WPRestrictedOpenerDomain : NSObject
@property (nonatomic, readonly) NSString *domain;
@property (nonatomic, readonly) WPRestrictedOpenerType openerType;
@end

typedef void (^WPRestrictedOpenerDomainsCompletionHandler)(NSArray<WPRestrictedOpenerDomain *> *, NSError *);

@interface WPResources (Staging_118208263)
- (void)requestRestrictedOpenerDomains:(WPResourceRequestOptions *)options completionHandler:(WPRestrictedOpenerDomainsCompletionHandler)completion;
@end
#endif

#if !defined(HAS_WEB_PRIVACY_STORAGE_ACCESS_PROMPT_TRIGGER) && HAVE(WEB_PRIVACY_FRAMEWORK)
@interface WPStorageAccessPromptQuirk (Staging_124689085)
@property (nonatomic, readonly) NSDictionary<NSString *, NSArray<NSString *> *> *quirkDomains;
@property (nonatomic, readonly) NSArray<NSString *> *triggerPages;
@end
#endif

#if !defined(HAS_WEB_PRIVACY_RESOURCE_MONITOR_URLS_API)
@class WKContentRuleList;
@class WKContentRuleListStore;

typedef void (^WPRuleListPreparationCompletionHandler)(WKContentRuleList *, bool, NSError *);

@interface WPResources (Staging_141646051)
- (void)prepareResouceMonitorRulesForStore:(WKContentRuleListStore *)store completionHandler:(WPRuleListPreparationCompletionHandler)completionHandler;
@end
#endif

WTF_EXTERN_C_BEGIN

extern NSString *const WPNotificationUserInfoResourceTypeKey;
extern NSNotificationName const WPResourceDataChangedNotificationName;

WTF_EXTERN_C_END

#if USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/WebPrivacySPIAdditions.h>)
#import <WebKitAdditions/WebPrivacySPIAdditions.h>
#endif

#endif // ENABLE(ADVANCED_PRIVACY_PROTECTIONS)
