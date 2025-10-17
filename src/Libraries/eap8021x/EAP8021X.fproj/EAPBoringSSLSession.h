/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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
typedef CFTypeRef EAPBoringSSLSessionContextRef;
typedef CFTypeRef EAPBoringSSLClientContextRef;

typedef enum {
    EAPBoringSSLSessionStateIdle,
    EAPBoringSSLSessionStateConnecting,
    EAPBoringSSLSessionStateConnected,
    EAPBoringSSLSessionStateDisconnected,
} EAPBoringSSLSessionState;

typedef OSStatus
(*EAPBoringSSLSessionReadFunc)(memoryIORef mem_io,
			       void *data, 		/* owned by* caller, data* RETURNED */
			       size_t *dataLength);	/* IN/OUT */

typedef OSStatus
(*EAPBoringSSLSessionWriteFunc)(memoryIORef mem_io,
				const void *data,
				size_t *dataLength);	/* IN/OUT */

typedef struct EAPBoringSSLSessionParameters_s {
    SecIdentityRef 			client_identity; 	/* SecIdentityRef */
    CFArrayRef 				client_certificates; 	/* Array of SecCertifictaeRef */
    tls_protocol_version_t		min_tls_version;
    tls_protocol_version_t 		max_tls_version;
    EAPBoringSSLSessionReadFunc 	read_func;
    EAPBoringSSLSessionWriteFunc 	write_func;
    EAPType 				eap_method;
    memoryIORef 			memIO;
} EAPBoringSSLSessionParameters, *EAPBoringSSLSessionParametersRef;


EAPBoringSSLSessionContextRef
EAPBoringSSLSessionContextCreate(EAPBoringSSLSessionParametersRef sessionParameters, EAPBoringSSLClientContextRef clientContext);

void
EAPBoringSSLSessionStart(EAPBoringSSLSessionContextRef sessionContext);

void
EAPBoringSSLSessionStop(EAPBoringSSLSessionContextRef sessionContext);

void
EAPBoringSSLSessionContextFree(EAPBoringSSLSessionContextRef sessionContext);

OSStatus
EAPBoringSSLSessionGetCurrentState(EAPBoringSSLSessionContextRef sessionContext, EAPBoringSSLSessionState *state);

CFStringRef
EAPBoringSSLSessionGetCurrentStateDescription(EAPBoringSSLSessionState state);

void
EAPBoringSSLUtilGetPreferredTLSVersions(CFDictionaryRef properties, tls_protocol_version_t *min, tls_protocol_version_t *max);

OSStatus
EAPBoringSSLSessionHandshake(EAPBoringSSLSessionContextRef sessionContext);

OSStatus
EAPBoringSSLSessionCopyServerCertificates(EAPBoringSSLSessionContextRef sessionContext, CFArrayRef *certs);

SecTrustRef
EAPBoringSSLSessionGetSecTrust(EAPBoringSSLSessionContextRef sessionContext);


OSStatus
EAPBoringSSLSessionComputeKeyData(EAPBoringSSLSessionContextRef sessionContext, void *key, int key_length);

OSStatus
EAPBoringSSLSessionGetNegotiatedTLSVersion(EAPBoringSSLSessionContextRef sessionContext, tls_protocol_version_t *tlsVersion);

OSStatus
EAPBoringSSLSessionGetSessionResumed(EAPBoringSSLSessionContextRef sessionContext, bool *sessionResumed);
