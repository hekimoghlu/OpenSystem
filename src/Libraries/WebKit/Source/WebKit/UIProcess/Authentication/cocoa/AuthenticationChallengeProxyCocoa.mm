/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#import "config.h"
#import "AuthenticationChallengeProxy.h"

#if HAVE(SEC_KEY_PROXY)

#import "ClientCertificateAuthenticationXPCConstants.h"
#import "Connection.h"
#import "SecKeyProxyStore.h"
#import <pal/spi/cocoa/NSXPCConnectionSPI.h>
#import <pal/spi/cocoa/SecKeyProxySPI.h>

namespace WebKit {

void AuthenticationChallengeProxy::sendClientCertificateCredentialOverXpc(IPC::Connection& connection, SecKeyProxyStore& secKeyProxyStore, AuthenticationChallengeIdentifier challengeID, const WebCore::Credential& credential)
{
    ASSERT(secKeyProxyStore.isInitialized());

    auto message = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_string(message.get(), ClientCertificateAuthentication::XPCMessageNameKey, ClientCertificateAuthentication::XPCMessageNameValue);
    xpc_dictionary_set_uint64(message.get(), ClientCertificateAuthentication::XPCChallengeIDKey, challengeID.toUInt64());
    xpc_dictionary_set_value(message.get(), ClientCertificateAuthentication::XPCSecKeyProxyEndpointKey, secKeyProxyStore.get().endpoint._endpoint);
    auto certificateDataArray = adoptOSObject(xpc_array_create(nullptr, 0));
    for (id certificate in credential.nsCredential().certificates) {
        auto data = adoptCF(SecCertificateCopyData((SecCertificateRef)certificate));
        xpc_array_append_value(certificateDataArray.get(), adoptOSObject(xpc_data_create(CFDataGetBytePtr(data.get()), CFDataGetLength(data.get()))).get());
    }
    xpc_dictionary_set_value(message.get(), ClientCertificateAuthentication::XPCCertificatesKey, certificateDataArray.get());
    xpc_dictionary_set_uint64(message.get(), ClientCertificateAuthentication::XPCPersistenceKey, static_cast<uint64_t>(credential.nsCredential().persistence));

    xpc_connection_send_message(connection.xpcConnection(), message.get());
}

} // namespace WebKit

#endif
