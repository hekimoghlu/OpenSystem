/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
 * pkcs12Derive.cpp - PKCS12 PBE routine
 *
 */
 
#ifndef	_PKCS12_DERIVE_H_
#define _PKCS12_DERIVE_H_

#include <Security/cssmtype.h>
#include <security_cdsa_utilities/context.h>
#include "AppleCSPSession.h"

#ifdef __cplusplus
extern "C" {
#endif

void DeriveKey_PKCS12 (
	const Context &context,
	AppleCSPSession	&session,
	const CssmData &Param,			// other's public key
	CSSM_DATA *keyData);			// mallocd by caller
									// we fill in keyData->Length bytes

#ifdef __cplusplus
}
#endif

#endif	/* _PKCS12_DERIVE_H_ */

