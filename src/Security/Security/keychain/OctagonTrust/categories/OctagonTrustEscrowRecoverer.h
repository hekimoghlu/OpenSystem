/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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

#ifndef OctagonTrustEscrowRecoverer_h
#define OctagonTrustEscrowRecoverer_h

NS_ASSUME_NONNULL_BEGIN

@protocol OctagonEscrowRecovererPrococol <NSObject>
- (NSError* _Nullable)recoverWithInfo:(NSDictionary* _Nullable)info results:(NSDictionary* _Nonnull* _Nullable)results;
- (NSError* _Nullable)getAccountInfoWithInfo:(NSDictionary* _Nullable)info results:(NSDictionary* _Nonnull* _Nullable)results;
- (NSError* _Nullable)disableWithInfo:(NSDictionary* _Nullable)info;
- (NSDictionary* _Nullable)recoverWithCDPContext:(OTICDPRecordContext*)cdpContext
                                    escrowRecord:(OTEscrowRecord*)escrowRecord
                                           error:(NSError**)error;
- (NSDictionary* _Nullable)recoverSilentWithCDPContext:(OTICDPRecordContext*)cdpContext
                                            allRecords:(NSArray<OTEscrowRecord*>*)allRecords
                                                 error:(NSError**)error;

- (NSDictionary* _Nullable)recoverWithCDPContext:(OTICDPRecordContext *)cdpContext
                                    escrowRecord:(OTEscrowRecord*)escrowRecord
                                         altDSID:(NSString* _Nullable)altDSID
                                          flowID:(NSString* _Nullable)flowID
                                 deviceSessionID:(NSString* _Nullable)deviceSessionID
                                           error:(NSError *__autoreleasing *)error;

- (NSDictionary* _Nullable)recoverSilentWithCDPContext:(OTICDPRecordContext*)cdpContext
                                            allRecords:(NSArray<OTEscrowRecord*>*)allRecords
                                               altDSID:(NSString* _Nullable)altDSID
                                                flowID:(NSString* _Nullable)flowID
                                       deviceSessionID:(NSString* _Nullable)deviceSessionID
                                                 error:(NSError**)error;

- (void)restoreKeychainAsyncWithPassword:password
                            keybagDigest:(NSData *)keybagDigest
                         haveBottledPeer:(BOOL)haveBottledPeer
                    viewsNotToBeRestored:(NSMutableSet <NSString*>*)viewsNotToBeRestored
                                   error:(NSError **)error;

- (bool)isRecoveryKeySet:(NSError**)error;

- (bool)restoreKeychainWithBackupPassword:(NSData *)password
                                    error:(NSError**)error;
- (NSError* _Nullable)backupWithInfo:(NSDictionary* _Nullable)info;
- (NSError* _Nullable)backupForRecoveryKeyWithInfo:(NSDictionary* _Nullable)info;

- (bool)verifyRecoveryKey:(NSString*)recoveryKey
                    error:(NSError**)error;

- (bool)removeRecoveryKeyFromBackup:(NSError**)error;
@end

NS_ASSUME_NONNULL_END

#endif /* OctagonTrustEscrowRecoverer_h */

#endif // OCTAGON
