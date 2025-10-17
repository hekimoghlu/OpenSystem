/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
#ifndef SecAppleAnchor_c
#define SecAppleAnchor_c

#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

typedef CF_OPTIONS(uint32_t, SecAppleTrustAnchorFlags) {
    kSecAppleTrustAnchorFlagsIncludeTestAnchors    = 1 << 0,
    kSecAppleTrustAnchorFlagsAllowNonProduction    = 1 << 1,
};

/*
 * Return true if the certificate is an Apple Trust anchor.
 */
bool
SecIsAppleTrustAnchor(SecCertificateRef cert,
                      SecAppleTrustAnchorFlags flags);

bool
SecIsAppleTrustAnchorData(CFDataRef cert,
			  SecAppleTrustAnchorFlags flags);

CFArrayRef SecGetAppleTrustAnchors(bool allowNonProduction);

/*
 * Return true if the certificate is an Apple Code Signing anchor.
 */
bool
SecIsAppleCodeSigningAnchor(SecCertificateRef cert);

/*
 * Return true if the issuer hash is an Apple Code Signing issuer.
 */
bool
SecIsAppleCodeSigningIssuer(CFDataRef issuerHash);

__END_DECLS


#endif /* SecAppleAnchor */
