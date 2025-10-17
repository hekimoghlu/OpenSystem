/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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
#ifndef _SECURITY_SECCERTIFICATESOURCE_H_
#define _SECURITY_SECCERTIFICATESOURCE_H_

#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecCertificate.h>
#include "trust/trustd/SecTrustServer.h"
#ifndef SecPVCRef
typedef struct OpaqueSecPVC *SecPVCRef;
#endif

/********************************************************
 ************ SecCertificateSource object ***************
 ********************************************************/
typedef struct SecCertificateSource *SecCertificateSourceRef;

typedef void(*SecCertificateSourceParents)(void *, CFArrayRef);

typedef bool(*CopyParents)(SecCertificateSourceRef source,
                           SecCertificateRef certificate,
                           void *context, SecCertificateSourceParents);

typedef CFArrayRef(*CopyConstraints)(SecCertificateSourceRef source,
                                     SecCertificateRef certificate);

typedef bool(*Contains)(SecCertificateSourceRef source,
                        SecCertificateRef certificate,
                        SecPVCRef pvc);

struct SecCertificateSource {
    CopyParents		copyParents;
    CopyConstraints	copyUsageConstraints;
    Contains		contains;
};

bool SecCertificateSourceCopyParents(SecCertificateSourceRef source,
                                     SecCertificateRef certificate,
                                     void *context, SecCertificateSourceParents callback);

CFArrayRef SecCertificateSourceCopyUsageConstraints(SecCertificateSourceRef source,
                                                    SecCertificateRef certificate);

bool SecCertificateSourceContains(SecCertificateSourceRef source,
                                  SecCertificateRef certificate,
                                  SecPVCRef pvc);

/********************************************************
 ********************** Sources *************************
 ********************************************************/

/* SecItemCertificateSource */
SecCertificateSourceRef SecItemCertificateSourceCreate(CFArrayRef accessGroups);
void SecItemCertificateSourceDestroy(SecCertificateSourceRef source);

/* SecMemoryCertificateSource*/
SecCertificateSourceRef SecMemoryCertificateSourceCreate(CFArrayRef certificates);
void SecMemoryCertificateSourceDestroy(SecCertificateSourceRef source);

/* SecSystemConstrainedAnchorSource */
bool SecSystemConstrainedAnchorSourceContainsAnchor(SecCertificateRef certificate);
CFArrayRef SecSystemConstrainedAnchorSourceCopyUsageConstraints(SecCertificateSourceRef source, SecCertificateRef certificate);
extern const SecCertificateSourceRef kSecSystemConstrainedAnchorSource;

/* SecSystemAnchorSource */
CFArrayRef SecSystemAnchorSourceCopyCertificates(void);
extern const SecCertificateSourceRef kSecSystemAnchorSource;

/* SecUserAnchorSource */
extern const SecCertificateSourceRef kSecUserAnchorSource;

/* SecCAIssuerCertificateSource */
extern const SecCertificateSourceRef kSecCAIssuerSource;

#if TARGET_OS_OSX
/* SecLegacyCertificateSource */
extern const SecCertificateSourceRef kSecLegacyCertificateSource;

/* SecLegacyAnchorSource */
extern const SecCertificateSourceRef kSecLegacyAnchorSource;
#endif

#endif /* _SECURITY_SECCERTIFICATESOURCE_H_ */
