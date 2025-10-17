/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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

/*! A WKSecurityOrigin object contains information about a security origin.
 @discussion An instance of this class is a transient, data-only object;
 it does not uniquely identify a security origin across multiple delegate method
 calls.
 */
NS_ASSUME_NONNULL_BEGIN

WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.11), ios(9.0))
@interface WKSecurityOrigin : NSObject

- (instancetype)init NS_UNAVAILABLE;

/*! @abstract The security origin's protocol.
 */
@property (nonatomic, readonly, copy) NSString *protocol;

/*! @abstract The security origin's host.
 */
@property (nonatomic, readonly, copy) NSString *host;

/*! @abstract The security origin's port.
 */
@property (nonatomic, readonly) NSInteger port;

@end

NS_ASSUME_NONNULL_END
