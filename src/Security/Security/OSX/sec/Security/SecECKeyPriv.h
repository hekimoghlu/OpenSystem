/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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
#ifndef _SECURITY_SECECKEYPRIV_H_
#define _SECURITY_SECECKEYPRIV_H_

#include <Security/SecECKey.h>
#include <corecrypto/ccec.h>

__BEGIN_DECLS

OSStatus SecECKeyGeneratePair(CFDictionaryRef parameters,
                              SecKeyRef *rsaPublicKey, SecKeyRef *rsaPrivateKey);

/* Vile accessors to enable stream encryption until
 we have better APIs for encryption with EC keys */
bool SecECDoWithFullKey(SecKeyRef key, CFErrorRef* error,void (^action)(ccec_full_ctx_t full_key));
bool SecECDoWithPubKey(SecKeyRef key, CFErrorRef* error, void (^action)(ccec_pub_ctx_t pub_key));

__END_DECLS

#endif /* !_SECURITY_SECECKEYPRIV_H_ */
