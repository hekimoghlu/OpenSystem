/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 22, 2024.
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

#import "keychain/ot/OTStashAccountSettingsOperation.h"

#import "utilities/debugging.h"
#import "keychain/ot/CuttlefishXPCWrapper.h"
#import "keychain/ot/OTOperationDependencies.h"
#import "keychain/ot/ObjCImprovements.h"
#import "keychain/ot/proto/generated_source/OTAccountSettings.h"
#import "keychain/ot/proto/generated_source/OTWalrus.h"
#import "keychain/ot/proto/generated_source/OTWebAccess.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperSpecificUser.h"

@interface OTStashAccountSettingsOperation ()
@property OTOperationDependencies* deps;
@property NSOperation* finishedOp;
@property id<OTAccountSettingsContainer> accountSettings;
@property bool accountWide;
@property bool forceFetch;
@end

@implementation OTStashAccountSettingsOperation
@synthesize intendedState = _intendedState;
@synthesize nextState = _nextState;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
                     accountSettings:(id<OTAccountSettingsContainer>)accountSettings
                         accountWide:(bool)accountWide
                          forceFetch:(bool)forceFetch
{
    if((self = [super init])) {
        _deps = dependencies;

        _intendedState = intendedState;
        _nextState = errorState;

        _accountSettings = accountSettings;
        _accountWide = accountWide;
        _forceFetch = forceFetch;
    }
    return self;
}

+ (void)performWithAccountWide:(bool)accountWide
                    forceFetch:(bool)forceFetch
          cuttlefishXPCWrapper:(CuttlefishXPCWrapper*)cuttlefishXPCWrapper
                 activeAccount:(TPSpecificUser* _Nullable)activeAccount
                 containerName:(NSString*)containerName
                     contextID:(NSString*)contextID
                         reply:(void (^)(OTAccountSettings* _Nullable settings, NSError* _Nullable error))reply
{
    if (accountWide) {
        [cuttlefishXPCWrapper fetchAccountSettingsWithSpecificUser:activeAccount
                                                        forceFetch:forceFetch
                                                             reply:^(NSDictionary<NSString*, TPPBPeerStableInfoSetting *> * _Nullable retSettings,
                                                                     NSError * _Nullable operror) {
                if(operror) {
                    secnotice("octagon", "Unable to fetch account settings for (%@,%@): %@", containerName, contextID, operror);
                    reply(nil, operror);
                } else {
                    if (retSettings && [retSettings count]) {
                        OTAccountSettings* settings = [[OTAccountSettings alloc] init];
                        OTWalrus* walrus = [[OTWalrus alloc]init];
                        if (retSettings[@"walrus"] != nil) {
                            TPPBPeerStableInfoSetting *walrusSetting = retSettings[@"walrus"];
                            walrus.enabled = walrusSetting.value;
                        }
                        settings.walrus = walrus;
                        OTWebAccess* webAccess = [[OTWebAccess alloc]init];
                        if (retSettings[@"webAccess"] != nil) {
                            TPPBPeerStableInfoSetting *webAccessSetting = retSettings[@"webAccess"];
                            webAccess.enabled = webAccessSetting.value;
                        }
                        settings.webAccess = webAccess;
                        reply(settings, nil);
                    } else {
                        reply(nil, [NSError errorWithDomain:OctagonErrorDomain code:OctagonErrorNoAccountSettingsSet userInfo: @{ NSLocalizedDescriptionKey : @"No account settings have been set"}]);
                    }
                }
            }];
    } else {
        [cuttlefishXPCWrapper fetchTrustStateWithSpecificUser:activeAccount
                                                        reply:^(TrustedPeersHelperPeerState * _Nullable selfPeerState,
                                                                NSArray<TrustedPeersHelperPeer *> * _Nullable trustedPeers,
                                                                NSError * _Nullable operror) {
                if(operror) {
                    secnotice("octagon", "Unable to fetch account settings for (%@,%@): %@", containerName, contextID, operror);
                    reply(nil, operror);
                } else {
                    OTAccountSettings* settings = [[OTAccountSettings alloc]init];
                    OTWalrus* walrus = [[OTWalrus alloc]init];
                    walrus.enabled = selfPeerState.walrus != nil ? selfPeerState.walrus.value : false;
                    settings.walrus = walrus;
                    OTWebAccess* webAccess = [[OTWebAccess alloc]init];
                    webAccess.enabled = selfPeerState.webAccess != nil ? selfPeerState.webAccess.value : true;
                    settings.webAccess = webAccess;
                    reply(settings, nil);
                }
            }];
    }
}

- (void)groupStart
{
    secnotice("octagon", "stashing account settings");

    self.finishedOp = [[NSOperation alloc] init];
    [self dependOnBeforeGroupFinished:self.finishedOp];

    WEAKIFY(self);
    [OTStashAccountSettingsOperation performWithAccountWide:self.accountWide
                                                 forceFetch:self.forceFetch
                                       cuttlefishXPCWrapper:self.deps.cuttlefishXPCWrapper
                                              activeAccount:self.deps.activeAccount
                                              containerName:self.deps.containerName
                                                  contextID:self.deps.contextID
                                                      reply:^(OTAccountSettings* _Nullable settings, NSError* _Nullable error) {
            STRONGIFY(self);
            NSError *stashError;
            [self.deps.stateHolder persistAccountChanges:^OTAccountMetadataClassC * _Nullable(OTAccountMetadataClassC * _Nonnull metadata) {
                    metadata.oldPeerID = metadata.peerID;
                    return metadata;
                } error:&stashError];
            if (stashError != nil) {
                self.error = stashError;
                [self.accountSettings setAccountSettings:nil];
            } else if (error != nil) {
                self.error = error;
                [self.accountSettings setAccountSettings:nil];
            } else {
                self.nextState = self.intendedState;
                [self.accountSettings setAccountSettings:settings];
            }
            [self runBeforeGroupFinished:self.finishedOp];
        }];
}

@end

#endif // OCTAGON
