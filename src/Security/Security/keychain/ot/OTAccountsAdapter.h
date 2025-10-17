/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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

#import <Foundation/Foundation.h>

#import "keychain/ot/OTPersonaAdapter.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperSpecificUser.h"

#import <AppleAccount/AppleAccount.h>

#import <AppleAccount/AppleAccount_Private.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#import <AppleAccount/ACAccount+AppleAccount.h>
#pragma clang diagnostic pop

NS_ASSUME_NONNULL_BEGIN

@protocol OTAccountsAdapter
- (TPSpecificUser* _Nullable)findAccountForCurrentThread:(id<OTPersonaAdapter>)personaAdapter
                                         optionalAltDSID:(NSString* _Nullable)altDSID
                                   cloudkitContainerName:(NSString*)cloudkitContainerName
                                        octagonContextID:(NSString*)octagonContextID
                                                   error:(NSError**)error;

- (NSArray<TPSpecificUser*>* _Nullable)inflateAllTPSpecificUsers:(NSString*)cloudkitContainerName
                                                octagonContextID:(NSString*)octagonContextID;

//test only
- (void)setAccountStore:(ACAccountStore*)store;

@end

@interface OTAccountsActualAdapter : NSObject <OTAccountsAdapter>
@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON
