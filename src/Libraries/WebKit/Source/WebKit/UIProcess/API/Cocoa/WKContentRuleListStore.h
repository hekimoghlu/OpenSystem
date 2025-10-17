/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#import <WebKit/WKFoundation.h>

#import <Foundation/Foundation.h>

@class WKContentRuleList;

WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.13), ios(11.0))
@interface WKContentRuleListStore : NSObject

+ (instancetype)defaultStore;
+ (instancetype)storeWithURL:(NSURL *)url;

- (void)compileContentRuleListForIdentifier:(NSString *)identifier encodedContentRuleList:(NSString *)encodedContentRuleList completionHandler:(void (^)(WKContentRuleList *, NSError *))completionHandler;

- (void)lookUpContentRuleListForIdentifier:(NSString *)identifier completionHandler:(void (^)(WKContentRuleList *, NSError *))completionHandler WK_SWIFT_ASYNC_NAME(contentRuleList(forIdentifier:));

- (void)removeContentRuleListForIdentifier:(NSString *)identifier completionHandler:(void (^)(NSError *))completionHandler;

- (void)getAvailableContentRuleListIdentifiers:(void (^)(NSArray<NSString *> *))completionHandler WK_SWIFT_ASYNC_NAME(availableIdentifiers());

@end
