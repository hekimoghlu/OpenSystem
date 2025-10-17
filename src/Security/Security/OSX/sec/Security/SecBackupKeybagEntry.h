/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
#import "CKKSSQLDatabaseObject.h"
#include <utilities/SecDb.h>
#include "keychain/securityd/SecDbItem.h"

#ifndef SecBackupKeybagEntry_h
#define SecBackupKeybagEntry_h

#if OCTAGON

@interface SecBackupKeybagEntry : CKKSSQLDatabaseObject {

}

//@property (getter=getChangeToken,setter=setChangeToken:) CKServerChangeToken* changeToken;
@property NSData* publickeyHash;
@property NSData* publickey;
@property NSData* musr;         // musr

+ (instancetype) state:(NSData*) publickeyHash;

+ (instancetype) fromDatabase: (NSData*) publickeyHash error: (NSError * __autoreleasing *) error;
+ (instancetype) tryFromDatabase: (NSData*) publickeyHash error: (NSError * __autoreleasing *) error;

- (instancetype) initWithPublicKey: (NSData*)publicKey publickeyHash: (NSData*) publickeyHash user: (NSData*) user;

- (BOOL)isEqual: (id) object;
@end

#endif
#endif /* SecBackupKeybagEntry_h */
