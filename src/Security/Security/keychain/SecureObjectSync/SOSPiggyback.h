/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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
#ifndef SOSCIRCLE_PIGGIGGYBACK_H
#define SOSCIRCLE_PIGGIGGYBACK_H 1

#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#include "keychain/SecureObjectSync/SOSCircle.h"

__BEGIN_DECLS

bool SOSPiggyBackBlobCreateFromData(SOSGenCountRef *gencount, SecKeyRef *pubKey, CFDataRef *signature,
                                    CFDataRef blobData, PiggyBackProtocolVersion version, bool *setInitialSyncTimeoutToV0, CFErrorRef *error);
bool SOSPiggyBackBlobCreateFromDER(SOSGenCountRef  *retGencount, SecKeyRef *retPubKey, CFDataRef *retSignature,
                                   const uint8_t** der_p, const uint8_t *der_end, PiggyBackProtocolVersion version, bool *setInitialSyncTimeoutToV0, CFErrorRef *error);
CFDataRef SOSPiggyBackBlobCopyEncodedData(SOSGenCountRef gencount, SecKeyRef pubKey, CFDataRef signature, CFErrorRef *error);

#if __OBJC__
bool SOSPiggyBackAddToKeychain(NSArray<NSData*>* identity, NSArray<NSDictionary*>*  tlk);
NSDictionary * SOSPiggyCopyInitialSyncData(const uint8_t** der, const uint8_t *der_end);
#endif

__END_DECLS

#endif /* SOSCIRCLE_PIGGIGGYBACK_H */
