/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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

enum OctagonSelfPeerKeysError: Error {
    case noPublicKeys
}

class OctagonSelfPeerKeys: NSObject, CKKSSelfPeer {
    var encryptionKey: _SFECKeyPair
    var signingKey: _SFECKeyPair
    var peerID: String

    // Here for conformance with CKKSPeer
    var publicEncryptionKey: _SFECPublicKey?
    var publicSigningKey: _SFECPublicKey?

    var encryptionVerificationKey: _SFECPublicKey
    var signingVerificationKey: _SFECPublicKey

    func matchesPeer(_ peer: CKKSPeer) -> Bool {
        return false
    }

    init(peerID: String, signingKey: _SFECKeyPair, encryptionKey: _SFECKeyPair) throws {
        self.peerID = peerID
        self.signingKey = signingKey
        self.encryptionKey = encryptionKey

        self.publicEncryptionKey = encryptionKey.publicKey as? _SFECPublicKey
        self.publicSigningKey = signingKey.publicKey as? _SFECPublicKey

        guard let encryptionVerificationKey = self.publicEncryptionKey,
            let signingVerificationKey = self.publicSigningKey else {
                throw OctagonSelfPeerKeysError.noPublicKeys
        }

        self.encryptionVerificationKey = encryptionVerificationKey
        self.signingVerificationKey = signingVerificationKey
    }

    override var description: String {
        return "<OctagonSelfPeerKeys: \(self.peerID)>"
    }
}

extension TPPeerPermanentInfo: @retroactive CKKSPeer {
    public var publicEncryptionKey: _SFECPublicKey? {
        return self.encryptionPubKey as? _SFECPublicKey
    }

    public var publicSigningKey: _SFECPublicKey? {
        return self.signingPubKey as? _SFECPublicKey
    }

    public func matchesPeer(_ peer: CKKSPeer) -> Bool {
        return self.peerID == peer.peerID
    }
}
