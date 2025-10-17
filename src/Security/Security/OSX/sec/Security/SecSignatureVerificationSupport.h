/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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

//
//  SecSignatureVerificationSupport.h
//
//

#ifndef _SECURITY_SECSIGNATUREVERIFICATION_H_
#define _SECURITY_SECSIGNATUREVERIFICATION_H_

#include <Availability.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecKey.h>
#include <Security/SecAsn1Types.h>
#include <libDER/DER_Keys.h>

bool SecVerifySignatureWithPublicKey(SecKeyRef publicKey, const DERAlgorithmId *sigAlgId,
                                     const uint8_t *dataToHash, size_t amountToHash,
                                     const uint8_t *signatureStart, size_t signatureSize,
                                     CFErrorRef *error)
    __OSX_AVAILABLE_STARTING(__MAC_10_12, __IPHONE_8_0);


#endif /* _SECURITY_SECSIGNATUREVERIFICATION_H_ */
