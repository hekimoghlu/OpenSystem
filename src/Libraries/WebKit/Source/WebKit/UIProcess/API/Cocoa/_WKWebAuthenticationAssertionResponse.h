/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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

NS_ASSUME_NONNULL_BEGIN

@class LAContext;

WK_CLASS_AVAILABLE(macos(11.0), ios(14.0))
@interface _WKWebAuthenticationAssertionResponse : NSObject

@property (nonatomic, readonly, copy) NSString *name;
@property (nonatomic, readonly, copy) NSString *displayName;
@property (nonatomic, readonly, nullable, copy) NSData *userHandle;
@property (nonatomic, readonly) BOOL synchronizable;
@property (nonatomic, readonly, copy) NSString *group;
@property (nonatomic, readonly, copy) NSData *credentialID;
@property (nonatomic, readonly, copy) NSString *accessGroup;

- (void)setLAContext:(LAContext *)context WK_API_AVAILABLE(macos(12.0), ios(15.0));

@end

NS_ASSUME_NONNULL_END
