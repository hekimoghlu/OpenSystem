/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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
import Foundation

class PolicyRedactionCrypter: NSObject, TPDecrypter, TPEncrypter {
    func decryptData(_ ciphertext: TPPBPolicyRedactionAuthenticatedCiphertext, withKey key: Data) throws -> Data {
        let operation = _SFAuthenticatedEncryptionOperation(keySpecifier: _SFAESKeySpecifier(bitSize: TPHObjectiveC.aes256BitSize()))
        let symmetricKey = try _SFAESKey(data: key, specifier: _SFAESKeySpecifier(bitSize: TPHObjectiveC.aes256BitSize()))

        let realCiphertext = _SFAuthenticatedCiphertext(ciphertext: ciphertext.ciphertext,
                                                        authenticationCode: ciphertext.authenticationCode,
                                                        initializationVector: ciphertext.initializationVector)

        let plaintext = try operation.decrypt(realCiphertext, with: symmetricKey)
        return plaintext
    }

    func encryptData(_ plaintext: Data, withKey key: Data) throws -> TPPBPolicyRedactionAuthenticatedCiphertext {
        let operation = _SFAuthenticatedEncryptionOperation(keySpecifier: _SFAESKeySpecifier(bitSize: TPHObjectiveC.aes256BitSize()))
        let symmetricKey = try _SFAESKey(data: key, specifier: _SFAESKeySpecifier(bitSize: TPHObjectiveC.aes256BitSize()))

        let ciphertext = try operation.encrypt(plaintext, with: symmetricKey, additionalAuthenticatedData: nil)

        let wrappedCiphertext = TPPBPolicyRedactionAuthenticatedCiphertext()!
        wrappedCiphertext.ciphertext = ciphertext.ciphertext
        wrappedCiphertext.authenticationCode = ciphertext.authenticationCode
        wrappedCiphertext.initializationVector = ciphertext.initializationVector

        return wrappedCiphertext
    }

    func randomKey() -> Data {
        var bytes = [Int8](repeating: 0, count: 256 / 8)
        guard errSecSuccess == SecRandomCopyBytes(kSecRandomDefault, 256 / 8, &bytes) else {
            abort()
        }

        return Data(bytes: bytes, count: 256 / 8)
    }
}
