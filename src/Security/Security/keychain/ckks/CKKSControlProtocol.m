/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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

#import "keychain/ckks/CKKSControlProtocol.h"

#if OCTAGON
#import <CloudKit/CloudKit.h>
#import <CloudKit/CloudKit_Private.h>
#import <objc/runtime.h>
#import "utilities/debugging.h"
#include <dlfcn.h>
#import <Security/SecXPCHelper.h>
#import <Security/CKKSExternalTLKClient.h>

// Weak-link CloudKit, until we can get ckksctl out of base system
static void *cloudKit = NULL;

static void
initCloudKit(void)
{
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        cloudKit = dlopen("/System/Library/Frameworks/CloudKit.framework/CloudKit", RTLD_LAZY);
    });
}

static void
getCloudKitSymbol(void **sym, const char *name)
{
    initCloudKit();
    if (!sym || *sym) {
        return;
    }
    *sym = dlsym(cloudKit, name);
    if (*sym == NULL) {
        fprintf(stderr, "symbol %s is missing", name);
        abort();
    }
}
#endif // OCTAGON

NSXPCInterface* CKKSSetupControlProtocol(NSXPCInterface* interface) {
#if OCTAGON
    static NSMutableSet *errClasses;

    static NSSet* tlkShareArrayClasses;
    static NSSet* tlkArrayClasses;

    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        __typeof(CKAcceptableValueClasses) *soft_CKAcceptableValueClasses = NULL;
        getCloudKitSymbol((void **)&soft_CKAcceptableValueClasses, "CKAcceptableValueClasses");
        errClasses = [NSMutableSet setWithSet:soft_CKAcceptableValueClasses()];
        [errClasses unionSet:[SecXPCHelper safeErrorClasses]];

        tlkArrayClasses = [NSSet setWithArray:@[[NSArray class], [CKKSExternalKey class]]];
        tlkShareArrayClasses = [NSSet setWithArray:@[[NSArray class], [CKKSExternalTLKShare class]]];
    });

    @try {
        [interface setClasses:errClasses forSelector:@selector(rpcResetLocal:reply:)                   argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(rpcResetCloudKit:reason:reply:)         argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(rpcResync:reply:)                       argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(rpcResyncLocal:reply:)                  argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(rpcStatus:fast:waitForNonTransientState:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(rpcFetchAndProcessChanges:classA:onlyIfNoRecentFetch:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(rpcPushOutgoingChanges:reply:)          argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(rpcGetCKDeviceIDWithReply:)             argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(rpcCKMetric:attributes:reply:)          argumentIndex:0 ofReply:YES];

        [interface setClasses:errClasses forSelector:@selector(proposeTLKForSEView:proposedTLK:wrappedOldTLK:tlkShares:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(fetchSEViewKeyHierarchy:forceFetch:reply:) argumentIndex:3 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(modifyTLKSharesForSEView:adding:deleting:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(deleteSEView:reply:) argumentIndex:0 ofReply:YES];

        [interface setClasses:errClasses forSelector:@selector(pcsMirrorKeysForServices:reply:) argumentIndex:1 ofReply:YES];

        [interface setClasses:tlkShareArrayClasses forSelector:@selector(proposeTLKForSEView:proposedTLK:wrappedOldTLK:tlkShares:reply:) argumentIndex:3 ofReply:NO];
        [interface setClasses:tlkArrayClasses      forSelector:@selector(fetchSEViewKeyHierarchy:forceFetch:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:tlkShareArrayClasses forSelector:@selector(fetchSEViewKeyHierarchy:forceFetch:reply:) argumentIndex:2 ofReply:YES];
        [interface setClasses:tlkShareArrayClasses forSelector:@selector(modifyTLKSharesForSEView:adding:deleting:reply:) argumentIndex:1 ofReply:NO];
        [interface setClasses:tlkShareArrayClasses forSelector:@selector(modifyTLKSharesForSEView:adding:deleting:reply:) argumentIndex:2 ofReply:NO];
    }

    @catch(NSException* e) {
        secerror("CKKSSetupControlProtocol failed, continuing, but you might crash later: %@", e);
        @throw e;
    }
#endif

    return interface;
}

