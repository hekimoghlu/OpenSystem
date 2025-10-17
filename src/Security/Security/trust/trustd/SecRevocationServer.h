/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 9, 2023.
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
/*!
 @header SecRevocationServer
 The functions provided in SecRevocationServer.h provide an interface to
 the trust evaluation engine for dealing with certificate revocation.
 */

#ifndef _SECURITY_SECREVOCATIONSERVER_H_
#define _SECURITY_SECREVOCATIONSERVER_H_

#include "trust/trustd/SecTrustServer.h"
#include "trust/trustd/SecRevocationDb.h"
#include "trust/trustd/SecOCSPRequest.h"
#include "trust/trustd/SecOCSPResponse.h"

typedef struct OpaqueSecORVC *SecORVCRef;

/* Revocation verification context. */
struct OpaqueSecRVC {
    /* Pointer to the builder for this revocation check */
    SecPathBuilderRef   builder;

    /* Index of cert in pvc that this RVC is for 0 = leaf, etc. */
    CFIndex             certIX;

    /* The OCSP Revocation verification context */
    SecORVCRef          orvc;

    /* Valid database info for this revocation check */
    SecValidInfoRef     valid_info;

    bool                done;

    bool                revocation_checked;
};
typedef struct OpaqueSecRVC *SecRVCRef;

CF_RETURNS_RETAINED SecORVCRef SecORVCCreate(SecRVCRef rvc, SecPathBuilderRef builder, CFIndex certIX);

/* OCSP Revocation verification context. */
struct OpaqueSecORVC {
    CFRuntimeBase _base;

    /* (weak) Pointer to the builder for this revocation check. */
    SecPathBuilderRef builder;

    /* Pointer to the generic rvc for this revocation check */
    SecRVCRef rvc;

    /* The ocsp request we send to each responder. */
    SecOCSPRequestRef ocspRequest;

    /* The freshest response we received so far, from stapling or cache or responder. */
    SecOCSPResponseRef ocspResponse;

    /* The best validated candidate single response we received so far, from stapling or cache or responder. */
    SecOCSPSingleResponseRef ocspSingleResponse;

    /* Index of cert in builder that this RVC is for 0 = leaf, etc. */
    CFIndex certIX;

    /* Validity period for which this revocation status. */
    CFAbsoluteTime thisUpdate;
    CFAbsoluteTime nextUpdate;

    /* URL of current responder. For logging purposes. */
    CFURLRef responder;

    bool done;
};

bool SecPathBuilderCheckRevocation(SecPathBuilderRef builder);
void SecPathBuilderCheckKnownIntermediateConstraints(SecPathBuilderRef builder);
CFAbsoluteTime SecRVCGetEarliestNextUpdate(SecRVCRef rvc);
CFAbsoluteTime SecRVCGetLatestThisUpdate(SecRVCRef rvc);
void SecRVCDelete(SecRVCRef rvc);
bool SecRVCHasDefinitiveValidInfo(SecRVCRef rvc);
bool SecRVCHasRevokedValidInfo(SecRVCRef rvc);
void SecRVCSetValidDeterminedErrorResult(SecRVCRef rvc);
bool SecRVCRevocationChecked(SecRVCRef rvc);

/* OCSP verification callbacks */
void SecORVCConsumeOCSPResponse(SecORVCRef rvc, SecOCSPResponseRef ocspResponse /*CF_CONSUMED*/,
                                CFTimeInterval maxAge, bool updateCache, bool fromCache);
void SecORVCUpdatePVC(SecORVCRef rvc);


#endif /* _SECURITY_SECREVOCATIONSERVER_H_ */
