/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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

extension SignedPeerPermanentInfo {
    init(_ permanentInfo: TPPeerPermanentInfo) {
        self.peerPermanentInfo = permanentInfo.data
        self.sig = permanentInfo.sig
    }

    func toPermanentInfo(peerID: String) -> TPPeerPermanentInfo? {
        return TPPeerPermanentInfo(peerID: peerID,
                                   data: self.peerPermanentInfo,
                                   sig: self.sig,
                                   keyFactory: TPECPublicKeyFactory())
    }
}

extension SignedPeerStableInfo {
    init(_ stableInfo: TPPeerStableInfo) {
        self.peerStableInfo = stableInfo.data
        self.sig = stableInfo.sig
    }

    func toStableInfo() -> TPPeerStableInfo? {
        return TPPeerStableInfo(data: self.peerStableInfo, sig: self.sig)
    }
}

extension SignedPeerDynamicInfo {
    init(_ dynamicInfo: TPPeerDynamicInfo) {
        self.peerDynamicInfo = dynamicInfo.data
        self.sig = dynamicInfo.sig
    }

    func toDynamicInfo() -> TPPeerDynamicInfo? {
        return TPPeerDynamicInfo(data: self.peerDynamicInfo, sig: self.sig)
    }
}

extension SignedCustodianRecoveryKey {
    init(_ crk: CustodianRecoveryKey) {
        self.custodianRecoveryKey = crk.tpCustodian.data
        self.sig = crk.tpCustodian.sig
    }

    func toCustodianRecoveryKey() -> TPCustodianRecoveryKey? {
        return TPCustodianRecoveryKey(data: self.custodianRecoveryKey,
                                      sig: self.sig,
                                      keyFactory: TPECPublicKeyFactory())
    }
}

extension TPPolicyVersion: @retroactive Comparable {
    public static func < (lhs: TPPolicyVersion, rhs: TPPolicyVersion) -> Bool {
        if lhs.versionNumber != rhs.versionNumber {
            return lhs.versionNumber < rhs.versionNumber
        } else {
            return lhs.policyHash < rhs.policyHash
        }
    }
}
