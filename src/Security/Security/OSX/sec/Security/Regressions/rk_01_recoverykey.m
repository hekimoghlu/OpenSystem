/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
//  rk_01_recoverykey.m
//

#include <Foundation/Foundation.h>
#include <Security/Security.h>

#include <Security/SecRecoveryKey.h>
#include "shared_regressions.h"

int rk_01_recoverykey(int argc, char *const *argv)
{
    NSArray *testData = @[
                          @{
                              @"recoverykey" : @"AAAA-AAAA-AAAA-AAAA-AAAA-AAAA-AAGW",
                              @"publicKey" : @"UUjq5Wv572RSsKahddvUPQAEIeErSHMK9J+NKb6sVdo=",
                              @"privateKey" : @"UUjq5Wv572RSsKahddvUPQAEIeErSHMK9J+NKb6sVdpi00pR5UGzfoARLnpxCFmqCh1XCRtjCptztGfN1XW11w==",
                              @"password" : @"Ze14tkzC8keZEnoIv+LoWvicxOTSSqUwhE8xyChmZAs=",
                              @"masterID" : @"Ndh2auLEx1zdQ+1KIAbpH4vbDN4L/O+C49AgAKhPGY0="
                              },
                          @{ // same again to make sure it works
                              @"recoverykey" : @"AAAA-AAAA-AAAA-AAAA-AAAA-AAAA-AAGW",
                              @"publicKey" : @"UUjq5Wv572RSsKahddvUPQAEIeErSHMK9J+NKb6sVdo=",
                              @"privateKey" : @"UUjq5Wv572RSsKahddvUPQAEIeErSHMK9J+NKb6sVdpi00pR5UGzfoARLnpxCFmqCh1XCRtjCptztGfN1XW11w==",
                              @"password" : @"Ze14tkzC8keZEnoIv+LoWvicxOTSSqUwhE8xyChmZAs=",
                              @"masterID" : @"Ndh2auLEx1zdQ+1KIAbpH4vbDN4L/O+C49AgAKhPGY0="

                              },
                          @{
                              @"recoverykey" : @"BBBB-BBBB-BBBB-BBBB-BBBB-BBBB-BBAY",
                              @"publicKey" : @"fomczHhXphIMaCbuQlKPefXO8YEIH2M9TFslcBjvJXY=",
                              @"privateKey" : @"fomczHhXphIMaCbuQlKPefXO8YEIH2M9TFslcBjvJXa/W5BWvgJmZO9xShq1sePpLDfGf5lOkwhwzFzFypiXgw==",
                              @"password" : @"P7nC1leKBTJ3aMsXZImVsR2kIlqlsvoSEI8yFKv6xdw=",
                              @"masterID" : @"aMGzcl99B0wBMQgKzSPCkpa8uG6g6As0+/aUgZOEpAI="
                              },
                          ];


    plan_tests(9 * (int)[testData count]);

    [testData enumerateObjectsUsingBlock:^(NSDictionary * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        NSString *recoveryKey = obj[@"recoverykey"];
        NSString *knownPublicKey = obj[@"publicKey"];
        NSString *knownPrivateKey = obj[@"privateKey"];
        NSString *knownPassword = obj[@"password"];
#if !(defined(__i386__) || TARGET_OS_SIMULATOR || TARGET_OS_BRIDGE)
        NSString *knownMasterID = obj[@"masterID"];
#endif

        SecRecoveryKey *rk = SecRKCreateRecoveryKeyWithError(recoveryKey, NULL);
        ok(rk, "got recovery key");

        NSData *publicKey = SecRKCopyBackupPublicKey(rk);
        ok(publicKey, "got publicKey");

        ok([publicKey isEqualToData:[[NSData alloc] initWithBase64EncodedString:knownPublicKey options:0]],
           "public key same: %@", [publicKey base64EncodedStringWithOptions:0]);

        NSData *privateKey = SecRKCopyBackupFullKey(rk);
        ok(privateKey, "got privateKey");

        ok([privateKey isEqualToData:[[NSData alloc] initWithBase64EncodedString:knownPrivateKey options:0]],
           "privateKey key same: %@", [privateKey base64EncodedStringWithOptions:0]);

        NSString *recoveryPassword = SecRKCopyAccountRecoveryPassword(rk);
        ok(recoveryPassword, "got account recovery password");

        ok([recoveryPassword isEqualToString:knownPassword], "password same: %@", recoveryPassword);
#if defined(__i386__) || TARGET_OS_SIMULATOR || TARGET_OS_BRIDGE
        ok(true, "skipping recovery verifier test for unsupported platform");
#else
        NSDictionary *recoveryVerifier = SecRKCopyAccountRecoveryVerifier(recoveryKey, NULL);
        NSData *masterID = [recoveryVerifier objectForKey:kSecRVMasterID];
        ok(masterID, "got masterID");
        ok([masterID isEqualToData:[[NSData alloc] initWithBase64EncodedString:knownMasterID options:0]],
           "masterID key same: %@", [masterID base64EncodedStringWithOptions:0]);

        // this can be uncommented to get some quick sample output
        // NSString *fileName = [NSString stringWithFormat:@"/tmp/%@.plist", recoveryKey];
        // [recoveryVerifier writeToFile:fileName atomically:YES];

#endif
    }];

    return 0;
}
