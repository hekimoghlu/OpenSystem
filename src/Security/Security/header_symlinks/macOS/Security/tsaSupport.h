/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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
#ifndef libsecurity_smime_tsaSupport_h
#define libsecurity_smime_tsaSupport_h

#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecCmsBase.h>

#if defined(__cplusplus)
extern "C" {
#endif

/*
 *   Time stamping Authority calls
 */

extern const CFStringRef kTSAContextKeyURL;      // CFURLRef
extern const CFStringRef kTSAContextKeyNoCerts;  // CFBooleanRef

OSStatus SecCmsTSADefaultCallback(CFTypeRef context,
                                  void* messageImprint,
                                  uint64_t nonce,
                                  CSSM_DATA* signedDERBlob) DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

CF_RETURNS_RETAINED CFMutableDictionaryRef SecCmsTSAGetDefaultContext(CFErrorRef* error);
void SecCmsMessageSetTSAContext(SecCmsMessageRef cmsg, CFTypeRef tsaContext);

#if defined(__cplusplus)
}
#endif

#endif /* libsecurity_smime_tsaSupport_h */
