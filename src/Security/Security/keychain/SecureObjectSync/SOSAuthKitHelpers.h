/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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

//
//  SOSAuthKitHelpers.h
//  Security
//

#ifndef SOSAuthKitHelpers_h
#define SOSAuthKitHelpers_h

#import "keychain/SecureObjectSync/SOSAccount.h"
#import "keychain/SecureObjectSync/SOSTrustedDeviceAttributes.h"

@interface SOSAuthKitHelpers : NSObject
+ (NSString * _Nullable)machineID;
+ (void) activeMIDs:(void(^_Nonnull)(NSSet <SOSTrustedDeviceAttributes *> * _Nullable activeMIDs, NSError * _Nullable error))complete;
+ (bool) updateMIDInPeerInfo: (SOSAccount *_Nonnull) account;
+ (bool) peerinfoHasMID: (SOSAccount *_Nonnull) account;
+ (bool) accountIsCDPCapable;
- (id _Nullable) initWithActiveMIDS: (NSSet *_Nullable) theMidList;
- (bool) midIsValidInList: (NSString *_Nullable) machineId;
- (bool) serialIsValidInList: (NSString *_Nullable) serialNumber;
- (bool) isUseful;

#if __OBJC2__

@property (nonatomic, retain) NSSet * _Nullable midList;
@property (nonatomic, retain) NSSet * _Nullable machineIDs;
@property (nonatomic, retain) NSSet * _Nullable serialNumbers;

#endif /* __OBJC2__ */

@end

#endif /* SOSAuthKitHelpers_h */
