/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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
#if OCTAGON

#import <utilities/debugging.h>

#import "keychain/ot/OTSetAccountSettingsOperation.h"
#import "keychain/ot/OTCuttlefishContext.h"
#import "keychain/ot/OTFetchCKKSKeysOperation.h"

#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/ObjCImprovements.h"
#import "keychain/categories/NSError+UsefulConstructors.h"

@interface OTSetAccountSettingsOperation ()
@property OTOperationDependencies* deps;

@property NSOperation* finishOp;
@end

@implementation OTSetAccountSettingsOperation
@synthesize nextState = _nextState;
@synthesize intendedState = _intendedState;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
                            settings:(OTAccountSettings* _Nullable)settings
{
    if((self = [super init])) {
        _deps = dependencies;
        _settings = settings;
        _deps = dependencies;
        _intendedState = intendedState;
        _nextState = errorState;
    }
    return self;
}

- (void)groupStart
{
    self.finishOp = [[NSOperation alloc] init];
    [self dependOnBeforeGroupFinished:self.finishOp];
    
    if (self.settings == nil) {
        self.nextState = self.intendedState;
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }

    TPPBPeerStableInfoSetting *walrus = nil;
    if (_settings.hasWalrus && _settings.walrus != nil) {
        walrus = [[TPPBPeerStableInfoSetting alloc]init];
        walrus.value = self.settings.walrus.enabled;
    }
    
    TPPBPeerStableInfoSetting *webAccess = nil;
    if (_settings.hasWebAccess && _settings.webAccess != nil) {
        webAccess = [[TPPBPeerStableInfoSetting alloc]init];
        webAccess.value = self.settings.webAccess.enabled;
    }
    
    WEAKIFY(self);
    
    [self.deps.cuttlefishXPCWrapper updateWithSpecificUser:self.deps.activeAccount
                                              forceRefetch:NO
                                                deviceName:nil
                                              serialNumber:nil
                                                 osVersion:nil
                                             policyVersion:nil
                                             policySecrets:nil
                                 syncUserControllableViews:nil
                                     secureElementIdentity:nil
                                             walrusSetting:walrus
                                                 webAccess:webAccess
                                                     reply:^(TrustedPeersHelperPeerState* peerState, TPSyncingPolicy* syncingPolicy, NSError* error) {
        STRONGIFY(self);
        TPPBPeerStableInfoSetting *walrus = peerState.walrus;
        TPPBPeerStableInfoSetting *webAccess = peerState.webAccess;
        NSError *walrusError = nil;
        NSError *webAccessError = nil;


        
        if (self.settings.walrus != nil && (walrus == nil || walrus.value != self.settings.walrus.enabled)) {
            secerror("octagon: error setting walrus: Intended value: %@, final value: %@, error: %@",
                     self.settings.walrus.enabled ? @"ON": @"OFF",
                     walrus == nil ? @"none" : walrus.value ? @"ON": @"OFF",
                     error);
            walrusError = [NSError errorWithDomain:OctagonErrorDomain
                                              code:OctagonErrorFailedToSetWalrus
                                       description:@"Failed to set walrus setting"
                                        underlying:error];
        }
        if (self.settings.webAccess != nil && (webAccess == nil || webAccess.value != self.settings.webAccess.enabled)) {
            secerror("octagon: Error setting web access: Intended value: %@, final value: %@, error: %@",
                     self.settings.webAccess.enabled ? @"ON": @"OFF",
                     webAccess.value ? @"ON": @"OFF",
                     error);
            webAccessError = [NSError errorWithDomain:OctagonErrorDomain
                                                 code:OctagonErrorFailedToSetWebAccess
                                          description:@"Failed to set web access setting"
                                           underlying:error];
        }
        if (walrusError && webAccessError) { //nest em
            walrusError = [NSError errorWithDomain:OctagonErrorDomain code:OctagonErrorFailedToSetWalrus description:@"Failed to set walrus setting" underlying:webAccessError];
            self.error = walrusError;
            [self runBeforeGroupFinished:self.finishOp];
            return;
        } else if (walrusError) {
            self.error = walrusError;
            [self runBeforeGroupFinished:self.finishOp];
            return;
        } else if (webAccessError) {
            self.error = webAccessError;
            [self runBeforeGroupFinished:self.finishOp];
            return;
        }
        self.nextState = self.intendedState;
        [self runBeforeGroupFinished:self.finishOp];
    }];
}

@end

#endif // OCTAGON
