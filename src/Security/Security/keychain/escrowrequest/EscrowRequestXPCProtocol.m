/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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
#import <objc/runtime.h>
#import <Security/SecXPCHelper.h>

#import "keychain/escrowrequest/EscrowRequestXPCProtocol.h"
#import "utilities/debugging.h"

NSXPCInterface* SecEscrowRequestSetupControlProtocol(NSXPCInterface* interface) {
    NSSet<Class>* errClasses = [SecXPCHelper safeErrorClasses];

    @try {
        [interface setClasses:errClasses forSelector:@selector(triggerEscrowUpdate:options:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(cachePrerecord:serializedPrerecord:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(fetchPrerecord:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(fetchRequestWaitingOnPasscode:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(fetchRequestStatuses:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(resetAllRequests:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(storePrerecordsInEscrow:) argumentIndex:1 ofReply:YES];
        
    }
    @catch(NSException* e) {
        secerror("SecEscrowRequestSetupControlProtocol failed, continuing, but you might crash later: %@", e);
        @throw e;
    }

    return interface;
}

