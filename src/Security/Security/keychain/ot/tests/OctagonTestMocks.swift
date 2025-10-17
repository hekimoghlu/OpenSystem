/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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
import Foundation

class OTMockSecureBackup: NSObject, OctagonEscrowRecovererPrococol {

    let bottleID: String?
    let entropy: Data?
    var recoveryKey: String?
    var kvsError: NSError?
    var sbdError: NSError?

    init(bottleID: String?, entropy: Data?) {
        self.bottleID = bottleID
        self.entropy = entropy
        self.recoveryKey = nil
        self.kvsError = nil
        self.sbdError = nil

        super.init()
    }

    func recover(withInfo info: [AnyHashable: Any]?,
                 results: AutoreleasingUnsafeMutablePointer<NSDictionary?>?) -> Error? {
        if self.bottleID == nil && self.entropy == nil {
            results!.pointee = [
                "bottleValid": "invalid",
            ]
        } else if self.bottleID == nil && self.entropy != nil {
            results!.pointee = [
                "EscrowServiceEscrowData": ["BottledPeerEntropy": self.entropy],
                "bottleValid": "invalid",
            ]
        } else if self.bottleID != nil && self.entropy == nil {
            results!.pointee = [
                "bottleID": self.bottleID!,
                "bottleValid": "invalid",
            ]
        } else { // entropy and bottleID must exist, so its a good bottle.
            results!.pointee = [
                "bottleID": self.bottleID!,
                "bottleValid": "valid",
                "EscrowServiceEscrowData": ["BottledPeerEntropy": self.entropy],
            ]
        }
        return nil
    }

    func disable(withInfo info: [AnyHashable: Any]!) -> Error? {
        return nil
    }

    func isRecoveryKeySet(_ error: NSErrorPointer) -> Bool {
        if self.kvsError != nil {
            error!.pointee = self.kvsError
            return false
        }
        return self.recoveryKey != nil ? true : false
    }

    func setRecoveryKey(recoveryKey: String?) {
        self.recoveryKey = recoveryKey
    }

    func restoreKeychain(withBackupPassword password: Data!, error: NSErrorPointer) -> Bool {
        if self.kvsError != nil {
            error!.pointee = self.kvsError
            return false
        }
        let RK = String(data: password!, encoding: String.Encoding.utf8)
        return RK == self.recoveryKey ? true : false
    }

    func verifyRecoveryKey(_ recoveryKey: String!, error: NSErrorPointer) -> Bool {
        if self.kvsError != nil {
            error!.pointee = self.kvsError
            return false
        }
        let result = (recoveryKey == self.recoveryKey ? true : false)
        if result == false {
            error!.pointee = NSError(domain: OctagonErrorDomain, code: OctagonError.recoveryKeyIncorrect.rawValue, userInfo: [NSLocalizedDescriptionKey: "Recovery key is incorrect"])
        }
        return result
    }

    func removeRecoveryKey(fromBackup error: NSErrorPointer) -> Bool {
        if self.sbdError != nil {
            error!.pointee = self.sbdError
            return false
        }
        self.recoveryKey = nil
        return true
    }

    func setExpectKVSError(_ error: NSError) {
        self.kvsError = error
    }

    func setExpectSBDError(_ error: NSError) {
        self.sbdError = error
    }

    @objc
    func getAccountInfo(withInfo info: [AnyHashable: Any]!, results: AutoreleasingUnsafeMutablePointer<NSDictionary?>!) -> Error? {
        let recordData = accountInfoWithInfoSample.data(using: .utf8)!
        var propertyListFormat = PropertyListSerialization.PropertyListFormat.xml
        do {
            results.pointee = try PropertyListSerialization.propertyList(from: recordData, options: .mutableContainersAndLeaves, format: &propertyListFormat) as! [String: AnyObject] as NSDictionary
        } catch {
            print("Error reading plist: \(error), format: \(propertyListFormat)")
        }
        return nil
    }
}

class OTMockFollowUpController: NSObject, OctagonFollowUpControllerProtocol {
    var postedFollowUp: Bool = false

    override init() {
        super.init()
    }

    func postFollowUp(with context: CDPFollowUpContext) throws {
        self.postedFollowUp = true
    }

    func clearFollowUp(with context: CDPFollowUpContext) throws {
    }
}

#endif // OCTAGON
