/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 21, 2022.
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
#ifndef libsecurity_smime_tsaSupport_priv_h
#define libsecurity_smime_tsaSupport_priv_h

#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecCmsBase.h>

#include <Security/SecAsn1Coder.h>
#include <Security/tsaTemplates.h>

#if defined(__cplusplus)
extern "C" {
#endif

extern const CFStringRef kTSADebugContextKeyBadReq;    // CFURLRef
extern const CFStringRef kTSADebugContextKeyBadNonce;  // CFBooleanRef

OSStatus SecTSAResponseCopyDEREncoding(SecAsn1CoderRef coder,
                                       const CSSM_DATA* tsaResponse,
                                       SecAsn1TimeStampRespDER* respDER);
OSStatus decodeTimeStampToken(SecCmsSignerInfoRef signerinfo,
                              CSSM_DATA_PTR inData,
                              CSSM_DATA_PTR encDigest,
                              uint64_t expectedNonce);
OSStatus decodeTimeStampTokenWithPolicy(SecCmsSignerInfoRef signerinfo,
                                        CFTypeRef timeStampPolicy,
                                        CSSM_DATA_PTR inData,
                                        CSSM_DATA_PTR encDigest,
                                        uint64_t expectedNonce);
OSStatus createTSAMessageImprint(SecCmsSignerInfoRef signerInfo,
                                 SECAlgorithmID* digestAlg,
                                 CSSM_DATA_PTR encDigest,
                                 SecAsn1TSAMessageImprint* messageImprint);

#ifndef NDEBUG
int tsaWriteFileX(const char* fileName, const unsigned char* bytes, size_t numBytes);
#endif

char* cfStringToChar(CFStringRef inStr);
uint64_t tsaDER_ToInt(const CSSM_DATA* DER_Data);
void displayTSTInfo(SecAsn1TSATSTInfo* tstInfo);

#if defined(__cplusplus)
}
#endif

#endif /* libsecurity_smime_tsaSupport_priv_h */
