/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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
// tokenaccess - access management to a TokenDatabase's Token's TokenDaemon's tokend
//
#include "tokenaccess.h"


//
// Process an exception thrown (presumably) by a TokenDaemon interface call.
//
void Access::operator () (const CssmError &err)
{
	if (++mIteration > 2) {
		secinfo("tokendb", "retry failed; aborting operation");
		throw;
	}
	
	//@@@ hack until tokend returns RESET
	if (err.error == -1) {
		secinfo("tokendb", "TEMP HACK (error -1) action - reset and retry");
		token.resetAcls();
		return;
	}
	
	if (CSSM_ERR_IS_CONVERTIBLE(err.error))
		switch (CSSM_ERRCODE(err.error)) {
		case CSSM_ERRCODE_OPERATION_AUTH_DENIED:
		case CSSM_ERRCODE_OBJECT_USE_AUTH_DENIED:
			// @@@ do something more focused here, but for now...
			secinfo("tokendb", "tokend denies auth; we're punting for now");
			throw;
		case CSSM_ERRCODE_DEVICE_RESET:
			secinfo("tokendb", "tokend signals reset; clearing and retrying");
			token.resetAcls();
			return;	// induce retry
		}
	// all others are non-recoverable
	secinfo("tokendb", "non-recoverable error in Access(): %d", err.error);
	throw;
}
