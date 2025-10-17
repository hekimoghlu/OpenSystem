/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
 @header SOSManifest.h
 The functions provided in SOSTransport.h provide an interface to the
 secure object syncing transport
 */

#ifndef _SEC_SOSMANIFEST_H_
#define _SEC_SOSMANIFEST_H_

#include <corecrypto/ccsha1.h>
#include <CoreFoundation/CFData.h>
#include <CoreFoundation/CFError.h>

__BEGIN_DECLS

enum {
    kSOSManifestUnsortedError = 1,
    kSOSManifestCreateError = 2,
};

extern CFStringRef kSOSManifestErrorDomain;

/* SOSObject. */

/* Forward declarations of SOS types. */
typedef struct __OpaqueSOSManifest *SOSManifestRef;
struct SOSDigestVector;

/* SOSManifest. */
CFTypeID SOSManifestGetTypeID(void);

SOSManifestRef SOSManifestCreateWithBytes(const uint8_t *bytes, size_t len,
                                          CFErrorRef *error);
SOSManifestRef SOSManifestCreateWithDigestVector(struct SOSDigestVector *dv, CFErrorRef *error);
SOSManifestRef SOSManifestCreateWithData(CFDataRef data, CFErrorRef *error);

size_t SOSManifestGetSize(SOSManifestRef m);

size_t SOSManifestGetCount(SOSManifestRef m);

const uint8_t *SOSManifestGetBytePtr(SOSManifestRef m);

CFDataRef SOSManifestGetData(SOSManifestRef m);

const struct SOSDigestVector *SOSManifestGetDigestVector(SOSManifestRef manifest);

bool SOSManifestDiff(SOSManifestRef a, SOSManifestRef b,
                     SOSManifestRef *a_minus_b, SOSManifestRef *b_minus_a,
                     CFErrorRef *error);

SOSManifestRef SOSManifestCreateWithPatch(SOSManifestRef base,
                                          SOSManifestRef removals,
                                          SOSManifestRef additions,
                                          CFErrorRef *error);

// Returns the set of elements in B that are not in A.
// This is the relative complement of A in B (B\A), sometimes written B âˆ’ A
SOSManifestRef SOSManifestCreateComplement(SOSManifestRef A,
                                           SOSManifestRef B,
                                           CFErrorRef *error);

SOSManifestRef SOSManifestCreateIntersection(SOSManifestRef m1,
                                             SOSManifestRef m2,
                                             CFErrorRef *error);


SOSManifestRef SOSManifestCreateUnion(SOSManifestRef m1,
                                      SOSManifestRef m2,
                                      CFErrorRef *error);

void SOSManifestForEach(SOSManifestRef m, void(^block)(CFDataRef e, bool *stop));

CFDataRef SOSManifestGetDigest(SOSManifestRef m, CFErrorRef *error);

__END_DECLS

#endif /* !_SEC_SOSMANIFEST_H_ */
