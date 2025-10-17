/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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
#import <WebKit/WKFoundation.h>

WK_CLASS_AVAILABLE(macos(10.15), ios(13.0))
@interface _WKContentRuleListAction : NSObject

@property (nonatomic, readonly) BOOL blockedLoad;
@property (nonatomic, readonly) BOOL blockedCookies;
@property (nonatomic, readonly) BOOL madeHTTPS;
@property (nonatomic, readonly) BOOL redirected WK_API_AVAILABLE(macos(13.0), ios(16.0));
@property (nonatomic, readonly) BOOL modifiedHeaders WK_API_AVAILABLE(macos(13.0), ios(16.0));
@property (nonatomic, readonly, copy) NSArray<NSString *> *notifications;

@end
