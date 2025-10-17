/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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

#import <IDS/IDS.h>
#import "OTPairingConstants.h"

NS_ASSUME_NONNULL_BEGIN

typedef void (^OTPairingCompletionHandler)(bool success, NSError *error);

@interface OTPairingService : NSObject <IDSServiceDelegate>

@property (readonly, nullable) NSString *pairedDeviceNotificationName;

+ (instancetype)sharedService;
- (instancetype)init NS_UNAVAILABLE;

- (void)initiatePairingWithCompletion:(OTPairingCompletionHandler)completionHandler;

@end

NS_ASSUME_NONNULL_END
