/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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
#import "SecAKSObjCWrappers.h"
#if __has_include(<UserManagement/UserManagement.h>)
#import <UserManagement/UserManagement.h>
#endif

@implementation SecAKSObjCWrappers

+ (bool)aksEncryptWithKeybag:(keybag_handle_t)keybag keyclass:(keyclass_t)keyclass plaintext:(NSData*)plaintext
                 outKeyclass:(keyclass_t*)outKeyclass ciphertext:(NSMutableData*)ciphertext personaId:(const void*)personaId personaIdLength:(size_t)personaIdLength error:(NSError**)error
{
    CFErrorRef cfError = NULL;
    bool result = false;
    if (personaId) {
        result = ks_crypt_diversify(kAKSKeyOpEncrypt, keybag, keyclass, (uint32_t)plaintext.length, plaintext.bytes, outKeyclass, (__bridge CFMutableDataRef)ciphertext, personaId, personaIdLength, &cfError);
    } else {
        result = ks_crypt(kAKSKeyOpEncrypt, keybag, NULL, keyclass, (uint32_t)plaintext.length, plaintext.bytes, outKeyclass, (__bridge CFMutableDataRef)ciphertext, false, &cfError);
    }
    BridgeCFErrorToNSErrorOut(error, cfError);
    return result;
}

+ (bool)aksDecryptWithKeybag:(keybag_handle_t)keybag keyclass:(keyclass_t)keyclass ciphertext:(NSData*)ciphertext
                 outKeyclass:(keyclass_t*)outKeyclass plaintext:(NSMutableData*)plaintext personaId:(const void*)personaId personaIdLength:(size_t)personaIdLength error:(NSError**)error
{
    CFErrorRef cfError = NULL;
    bool result = false;
    if (personaId) {
        result = ks_crypt_diversify(kAKSKeyOpDecrypt, keybag, keyclass, (uint32_t)ciphertext.length, ciphertext.bytes, outKeyclass, (__bridge CFMutableDataRef)plaintext, personaId, personaIdLength, &cfError);
    } else {
        result = ks_crypt(kAKSKeyOpDecrypt, keybag, NULL, keyclass, (uint32_t)ciphertext.length, ciphertext.bytes, outKeyclass, (__bridge CFMutableDataRef)plaintext, false, &cfError);
    }
    BridgeCFErrorToNSErrorOut(error, cfError);
    return result;
}

@end
