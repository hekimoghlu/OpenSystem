/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 6, 2022.
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

// server.c undefs KCSHARING on darwinOS / system securityd, we want to match those conditions here.
#if !KCSHARING || (defined(TARGET_DARWINOS) && TARGET_DARWINOS) || (defined(SECURITYD_SYSTEM) && SECURITYD_SYSTEM)

#import "KCSharingStubXPCListenerDelegate.h"

void KCSharingStubXPCServerInitialize(void) {
    [KCSharingStubXPCListenerDelegate sharedInstance];
}

@implementation KCSharingStubXPCListenerDelegate {
    NSXPCListener* _listener;
}

+ (instancetype)sharedInstance {
    static dispatch_once_t once;
    static KCSharingStubXPCListenerDelegate *delegate;

    dispatch_once(&once, ^{
        @autoreleasepool {
            delegate = [[KCSharingStubXPCListenerDelegate alloc] init];
        }
    });

    return delegate;
}

- (instancetype)init {
    if (self = [super init]) {
        _listener = [[NSXPCListener alloc] initWithMachServiceName:@"com.apple.security.kcsharing"];
        _listener.delegate = self;
        [_listener activate];
    }
    return self;
}

- (BOOL)listener:(NSXPCListener *)listener shouldAcceptNewConnection:(NSXPCConnection *)newConnection {
    return NO;
}

@end

#endif /* !KCSHARING || (defined(TARGET_DARWINOS) && TARGET_DARWINOS) || (defined(SECURITYD_SYSTEM) && SECURITYD_SYSTEM) */
