/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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

/*
 This header and its implementation are here because keychain/Sharing is excluded entirely for !KCSHARING.
 We want this so that we can install the mach service unconditionally and have a dummy service check-in at runtime
 if the feature itself is absent, which in turn is helpful for development.
 */

// server.c undefs KCSHARING on darwinOS / system securityd, we want to match those conditions here.
#if !KCSHARING || (defined(TARGET_DARWINOS) && TARGET_DARWINOS) || (defined(SECURITYD_SYSTEM) && SECURITYD_SYSTEM)

void KCSharingStubXPCServerInitialize(void);

#if __OBJC__

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface KCSharingStubXPCListenerDelegate : NSObject <NSXPCListenerDelegate>

+ (instancetype)sharedInstance;

@end

NS_ASSUME_NONNULL_END

#endif /* __OBJC__ */
#endif /* !KCSHARING || (defined(TARGET_DARWINOS) && TARGET_DARWINOS) || (defined(SECURITYD_SYSTEM) && SECURITYD_SYSTEM) */
