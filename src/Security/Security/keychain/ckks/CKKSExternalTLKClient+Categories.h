/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
#import <CloudKit/CloudKit.h>

#import "keychain/ckks/CKKSExternalTLKClient.h"
#import "keychain/ckks/CKKSKey.h"
#import "keychain/ckks/CKKSTLKShare.h"
#import "keychain/ckks/CKKSTLKShareRecord.h"

NS_ASSUME_NONNULL_BEGIN

@interface CKKSExternalKey (CKKSTranslation)
- (instancetype)initWithViewName:(NSString*)viewName
                             tlk:(CKKSKey*)tlk;

- (CKKSKey* _Nullable)makeCKKSKey:(CKRecordZoneID*)zoneID
                        contextID:(NSString*)contextID
                            error:(NSError**)error;

// The CKKS cloudkit plugin ensures that there is a classA and classC 'key' in the key hierarchy. Fake it.
- (CKKSKey* _Nullable)makeFakeCKKSClassKey:(CKKSKeyClass*)keyclass
                                 contextID:(NSString*)contextID
                                    zoneiD:(CKRecordZoneID*)zoneID
                                     error:(NSError**)error;
@end

@interface CKKSExternalTLKShare (CKKSTranslation)
- (instancetype)initWithViewName:(NSString*)viewName
                        tlkShare:(CKKSTLKShare*)share;

- (CKKSTLKShareRecord* _Nullable)makeTLKShareRecord:(CKRecordZoneID*)zoneID
                                          contextID:(NSString*)contextID;
@end

NS_ASSUME_NONNULL_END

#endif
