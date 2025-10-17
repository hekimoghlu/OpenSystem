/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#import "AuthenticationManager.h"

#if HAVE(SEC_KEY_PROXY)

#import "AuthenticationChallengeDisposition.h"
#import "ClientCertificateAuthenticationXPCConstants.h"
#import "Connection.h"
#import "XPCUtilities.h"
#import <WebCore/Credential.h>
#import <pal/spi/cocoa/NSXPCConnectionSPI.h>
#import <pal/spi/cocoa/SecKeyProxySPI.h>
#import <wtf/MainThread.h>

namespace WebKit {

void AuthenticationManager::initializeConnection(IPC::Connection* connection)
{
    RELEASE_ASSERT(isMainRunLoop());

    if (!connection || !connection->xpcConnection()) {
        ASSERT_NOT_REACHED();
        return;
    }

    WeakPtr weakThis { *this };
    // The following xpc event handler overwrites the boostrap event handler and is only used
    // to capture client certificate credential.
    xpc_connection_set_event_handler(connection->xpcConnection(), ^(xpc_object_t event) {
#if USE(EXIT_XPC_MESSAGE_WORKAROUND)
        handleXPCExitMessage(event);
#endif
        callOnMainRunLoop([event = OSObjectPtr(event), weakThis = weakThis] {
            RELEASE_ASSERT(isMainRunLoop());

            xpc_type_t type = xpc_get_type(event.get());
            if (type == XPC_TYPE_ERROR || !weakThis)
                return;

            if (type != XPC_TYPE_DICTIONARY || xpc_dictionary_get_wtfstring(event.get(), ClientCertificateAuthentication::XPCMessageNameKey) != ClientCertificateAuthentication::XPCMessageNameValue) {
                ASSERT_NOT_REACHED();
                return;
            }

            auto challengeID = xpc_dictionary_get_uint64(event.get(), ClientCertificateAuthentication::XPCChallengeIDKey);
            if (!challengeID)
                return;

            auto xpcEndPoint = xpc_dictionary_get_value(event.get(), ClientCertificateAuthentication::XPCSecKeyProxyEndpointKey);
            if (!xpcEndPoint || xpc_get_type(xpcEndPoint) != XPC_TYPE_ENDPOINT)
                return;
            auto endPoint = adoptNS([[NSXPCListenerEndpoint alloc] init]);
            [endPoint _setEndpoint:xpcEndPoint];
            NSError *error = nil;
            auto identity = adoptCF([SecKeyProxy createIdentityFromEndpoint:endPoint.get() error:&error]);
            if (!identity || error) {
                LOG_ERROR("Couldn't create identity from end point: %@", error);
                return;
            }

            auto certificateDataArray = xpc_dictionary_get_array(event.get(), ClientCertificateAuthentication::XPCCertificatesKey);
            if (!certificateDataArray)
                return;
            NSMutableArray *certificates = nil;
            if (auto total = xpc_array_get_count(certificateDataArray)) {
                certificates = [NSMutableArray arrayWithCapacity:total];
                for (size_t i = 0; i < total; i++) {
                    auto certificateData = xpc_array_get_value(certificateDataArray, i);
                    auto cfData = adoptCF(CFDataCreate(nullptr, static_cast<const UInt8*>(xpc_data_get_bytes_ptr(certificateData)), xpc_data_get_length(certificateData)));
                    auto certificate = adoptCF(SecCertificateCreateWithData(nullptr, cfData.get()));
                    if (!certificate)
                        return;
                    [certificates addObject:(__bridge id)certificate.get()];
                }
            }

            auto persistence = xpc_dictionary_get_uint64(event.get(), ClientCertificateAuthentication::XPCPersistenceKey);
            if (persistence > static_cast<uint64_t>(NSURLCredentialPersistenceSynchronizable))
                return;

            weakThis->completeAuthenticationChallenge(ObjectIdentifier<AuthenticationChallengeIdentifierType>(challengeID), AuthenticationChallengeDisposition::UseCredential, WebCore::Credential(adoptNS([[NSURLCredential alloc] initWithIdentity:identity.get() certificates:certificates persistence:(NSURLCredentialPersistence)persistence]).get()));
        });
    });
}

} // namespace WebKit

#endif
