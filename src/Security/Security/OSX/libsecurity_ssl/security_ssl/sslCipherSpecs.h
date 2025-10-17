/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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
/*
 * cipherSpecs.h - SSLCipherSpec declarations
 */

#ifndef	_SSL_CIPHER_SPECS_H_
#define _SSL_CIPHER_SPECS_H_

#include "sslContext.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*
 * Build ctx->validCipherSuites as a copy of all known CipherSpecs.
 */
OSStatus sslBuildCipherSuiteArray(SSLContext *ctx);

/*
 * Initialize dst based on selectedCipher.
 */
void InitCipherSpecParams(SSLContext *ctx);

/*
 * Given a valid ctx->selectedCipher and ctx->validCipherSuites, set
 * ctx->selectedCipherSpec as appropriate. Return an error if
 * ctx->selectedCipher could not be set as the current ctx->selectedCipherSpec.
 */
OSStatus FindCipherSpec(SSLContext *ctx);

#ifdef __cplusplus
}
#endif

#endif	/* _CIPHER_SPECS_H_ */
