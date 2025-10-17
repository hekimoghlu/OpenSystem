/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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

class ProxyXPCConnection: NSObject, NSXPCListenerDelegate {
    let obj: Any
    let serverInterface: NSXPCInterface
    let listener: NSXPCListener

    init(_ obj: Any, interface: NSXPCInterface) {
        self.obj = obj
        self.serverInterface = interface
        self.listener = NSXPCListener.anonymous()

        super.init()
        self.listener.delegate = self
        self.listener.resume()
    }

    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection newConnection: NSXPCConnection) -> Bool {
        newConnection.exportedInterface = self.serverInterface
        newConnection.exportedObject = self.obj
        newConnection.resume()
        return true
    }

    func connection() -> NSXPCConnection {
        let connection = NSXPCConnection(listenerEndpoint: self.listener.endpoint)
        connection.remoteObjectInterface = self.serverInterface
        connection.resume()
        return connection
    }
}

class FakeNSXPCConnectionSOS: NSXPCConnection {
    var sosControl: SOSControlProtocol

    init(withSOSControl: SOSControlProtocol) {
        self.sosControl = withSOSControl
        super.init()
    }

    override func remoteObjectProxyWithErrorHandler(_ handler: @escaping (Error) -> Void) -> Any {
        return self.sosControl
    }

    override func synchronousRemoteObjectProxyWithErrorHandler(_ handler: @escaping (Error) -> Void) -> Any {
        return FakeNSXPCConnection(control: self.sosControl)
    }
}

class FakeOTControlEntitlementBearer: OctagonEntitlementBearerProtocol {
    var entitlements: [String: Any]

    init() {
        // By default, this client has all octagon entitlements
        self.entitlements = [kSecEntitlementPrivateOctagonEscrow: true]
    }
    func value(forEntitlement entitlement: String) -> Any? {
        return self.entitlements[entitlement]
    }
}

#endif // OCTAGON
