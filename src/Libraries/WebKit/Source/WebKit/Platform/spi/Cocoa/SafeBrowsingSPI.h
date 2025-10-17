/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
#if HAVE(SAFE_BROWSING)

#import <Foundation/Foundation.h>

#if 0 && USE(APPLE_INTERNAL_SDK)

#import <SafariSafeBrowsing/SafariSafeBrowsing.h>

#else

typedef NSString * SSBProvider NS_STRING_ENUM;

WTF_EXTERN_C_BEGIN

extern SSBProvider const SSBProviderGoogle;
extern SSBProvider const SSBProviderTencent;

WTF_EXTERN_C_END

@interface SSBServiceLookupResult : NSObject <NSCopying, NSSecureCoding>

@property (nonatomic, readonly) SSBProvider provider;

@property (nonatomic, readonly, getter=isPhishing) BOOL phishing;
@property (nonatomic, readonly, getter=isMalware) BOOL malware;
@property (nonatomic, readonly, getter=isUnwantedSoftware) BOOL unwantedSoftware;

#if HAVE(SAFE_BROWSING_RESULT_DETAILS)
@property (nonatomic, readonly) NSString *malwareDetailsBaseURLString;
@property (nonatomic, readonly) NSURL *learnMoreURL;
@property (nonatomic, readonly) NSString *reportAnErrorBaseURLString;
@property (nonatomic, readonly) NSString *localizedProviderDisplayName;
#endif

@end

@interface SSBLookupResult : NSObject <NSCopying, NSSecureCoding>

@property (nonatomic, readonly) NSArray<SSBServiceLookupResult *> *serviceLookupResults;

@end

@interface SSBLookupContext : NSObject

+ (SSBLookupContext *)sharedLookupContext;

- (void)lookUpURL:(NSURL *)URL completionHandler:(void (^)(SSBLookupResult *, NSError *))completionHandler;

@end

#endif

#endif
