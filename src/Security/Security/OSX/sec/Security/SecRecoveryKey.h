/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
//  SecRecoveryKey.h
//
//

#ifndef SecRecoveryKey_h
#define SecRecoveryKey_h

#include <Security/Security.h>
#if __OBJC__
@class SecRecoveryKey;
#else
typedef struct __SecRecoveryKey SecRecoveryKey;
#endif

bool
SecRKRegisterBackupPublicKey(SecRecoveryKey *rk, CFErrorRef *error);

#if __OBJC__

/*
 * Constants for the verifier dictionary returned from SecRKCopyAccountRecoveryVerifier
 */

extern NSString *const kSecRVSalt;
extern NSString *const kSecRVIterations;
extern NSString *const kSecRVProtocol;
extern NSString *const kSecRVVerifier;
extern NSString *const kSecRVMasterID;


SecRecoveryKey *
SecRKCreateRecoveryKey(NSString *recoveryKey);

SecRecoveryKey *
SecRKCreateRecoveryKeyWithError(NSString *masterKey, NSError **error);

NSString *
SecRKCreateRecoveryKeyString(NSError **error);

NSString *
SecRKCopyAccountRecoveryPassword(SecRecoveryKey *rk);

NSData *
SecRKCopyBackupFullKey(SecRecoveryKey *rk);

NSData *
SecRKCopyBackupPublicKey(SecRecoveryKey *rk);

NSDictionary *
SecRKCopyAccountRecoveryVerifier(NSString *recoveryKey,
                                 NSError **error);

#else

SecRecoveryKey *
SecRKCreateRecoveryKey(CFStringRef recoveryKey);

CFDataRef
SecRKCopyBackupFullKey(SecRecoveryKey *rk);

CFDataRef
SecRKCopyBackupPublicKey(SecRecoveryKey *rk);

#endif

#endif
