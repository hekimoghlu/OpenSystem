/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
#import "krb5.h"
#import <os/log.h>

NS_ASSUME_NONNULL_BEGIN

os_log_t GSSHelperOSLog(void);

@interface GSSHelperPeer : NSObject

@property (nonatomic) xpc_connection_t conn;
@property (nonatomic) NSString *bundleIdentifier;
@property (nonatomic) uid_t session;

@end

//This is called by GSSCred when a session is joined from itself
@interface GSSCredHelper : NSObject

+ (void)do_Acquire:(GSSHelperPeer *)peer request:(xpc_object_t) request reply:(xpc_object_t) reply;
+ (void)do_Refresh:(GSSHelperPeer *)peer request:(xpc_object_t) request reply:(xpc_object_t) reply;

@end

NS_ASSUME_NONNULL_END
