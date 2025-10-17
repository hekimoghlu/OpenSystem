/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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
typedef struct __EAPTLSSessionContext * EAPTLSSessionContextRef;

EAPTLSSessionContextRef
EAPTLSSessionCreateContext(CFDictionaryRef properties, EAPType eapType, memoryIORef mem_io, CFArrayRef client_certificates, OSStatus * ret_status);

void
EAPTLSSessionFreeContext(EAPTLSSessionContextRef session_context);

OSStatus
EAPTLSSessionClose(EAPTLSSessionContextRef session_context);

OSStatus
EAPTLSSessionSetPeerID(EAPTLSSessionContextRef session_context, const void *peer_id, size_t peer_id_len);

OSStatus
EAPTLSSessionGetState(EAPTLSSessionContextRef session_context, SSLSessionState *state);

OSStatus
EAPTLSSessionHandshake(EAPTLSSessionContextRef session_context);

void
EAPTLSSessionCopyPeerCertificates(EAPTLSSessionContextRef session_context, CFArrayRef * peer_certificates);

SecTrustRef
EAPTLSSessionGetSecTrust(EAPTLSSessionContextRef session_context);

Boolean
EAPTLSSessionIsRevocationStatusCheckRequired(EAPTLSSessionContextRef session_context);

OSStatus
EAPTLSSessionComputeSessionKey(EAPTLSSessionContextRef session_context, const void * label, int label_length, void * key, int key_length);

void
EAPTLSSessionGetSessionResumed(EAPTLSSessionContextRef session_context, Boolean *resumed);

CFStringRef
EAPTLSSessionGetNegotiatedTLSProtocolVersion(EAPTLSSessionContextRef session_context);

void
EAPTLSSessionGetNegotiatedCipher(EAPTLSSessionContextRef session_context, SSLCipherSuite *cipher_suite);
