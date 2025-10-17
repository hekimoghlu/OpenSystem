/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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
import Foundation_Private.NSXPCConnection
import os.log

setupICUMallocZone()

let containerMap = ContainerMap(ckCodeOperationRunnerCreator: CuttlefishCKOperationRunnerCreator(),
                                darwinNotifier: CKKSNotifyPostNotifier.self,
                                personaAdapter: OTPersonaActualAdapter(),
                                managedConfigurationAdapter: OTManagedConfigurationActualAdapter())

private let logger = Logger(subsystem: "com.apple.security.trustedpeers", category: "main")

class ServiceDelegate: NSObject, NSXPCListenerDelegate {
    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection newConnection: NSXPCConnection) -> Bool {
        let tphEntitlement = "com.apple.private.trustedpeershelper.client"

        logger.info("Received a new client: \(newConnection, privacy: .public)")

#if os(macOS)
        // The recommended way to do this check is: SAUserSetupState.getForUser(getuid()) != .setupUser
        // That call is expensive in the non-setup case; use constant from MacBuddyX/SharedConstants.h
        let kMBBuddyUserID = 248
        guard getuid() != kMBBuddyUserID else {
            logger.info("client(\(newConnection, privacy: .public)) is running as setup user")
            return false
        }
#endif

        switch newConnection.value(forEntitlement: tphEntitlement) {
        case 1 as Int:
            logger.info("client has entitlement '\(tphEntitlement, privacy: .public)'")
        case true as Bool:
            logger.info("client has entitlement '\(tphEntitlement, privacy: .public)'")

        case let someInt as Int:
            logger.info("client(\(newConnection, privacy: .public) has wrong integer value for '\(tphEntitlement, privacy: .public)' (\(someInt)), rejecting")
            return false

        case let someBool as Bool:
            logger.info("client(\(newConnection, privacy: .public) has wrong boolean value for '\(tphEntitlement, privacy: .public)' (\(someBool)), rejecting")
            return false

        default:
            logger.info("client(\(newConnection, privacy: .public) is missing entitlement '\(tphEntitlement, privacy: .public)'")
            return false
        }

        newConnection.exportedInterface = TrustedPeersHelperSetupProtocol(NSXPCInterface(with: TrustedPeersHelperProtocol.self))
        let exportedObject = Client(endpoint: newConnection.endpoint, containerMap: containerMap)
        newConnection.exportedObject = exportedObject
        newConnection.resume()

        return true
    }
}

let sandboxIdentifier = "com.apple.TrustedPeersHelper"

#if os(macOS)
    Sandbox.enterSandbox(identifier: sandboxIdentifier, profile: sandboxIdentifier)
#else
    Sandbox.enterSandbox(identifier: sandboxIdentifier)
#endif

logger.info("Starting up")

ValueTransformer.setValueTransformer(SetValueTransformer(), forName: SetValueTransformer.name)

let delegate = ServiceDelegate()
let listener = NSXPCListener.service()

listener.delegate = delegate
listener.resume()
