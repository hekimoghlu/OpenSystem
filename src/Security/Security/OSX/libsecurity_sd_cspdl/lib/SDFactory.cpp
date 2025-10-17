/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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
// SDFactory -- The factory for Security Server context objects
//
#include "SDFactory.h"

#include "SDContext.h"


//
// SDFactory -- The factory for Security Server context objects
//
bool SDFactory::setup(SDCSPSession &session, CSPFullPluginSession::CSPContext * &cspCtx,
					  const Context &context, bool encoding)
{
	if (cspCtx)
		return false;	// not ours or already set

	switch (context.type())
	{
	case CSSM_ALGCLASS_SIGNATURE:
		cspCtx = new SDSignatureContext(session);
		return true;
	case CSSM_ALGCLASS_MAC:
		cspCtx = new SDMACContext(session);
		return true;
	case CSSM_ALGCLASS_DIGEST:
		cspCtx = new SDDigestContext(session);
		return true;
	case CSSM_ALGCLASS_SYMMETRIC:
	case CSSM_ALGCLASS_ASYMMETRIC:
		cspCtx = new SDCryptContext(session); // @@@ Could also be wrap/unwrap
		return true;
	case CSSM_ALGCLASS_RANDOMGEN:
		cspCtx = new SDRandomContext(session); // @@@ Should go.
		return true;
	}

	return false;

#if 0
	/* FIXME - qualify by ALGCLASS as well to avoid MAC */
	switch (context.algorithm()) {
	case CSSM_ALGID_MD5:
		cspCtx = new MD5Context(session);
		return true;
	case CSSM_ALGID_SHA1:
		cspCtx = new SHA1Context(session);
		return true;
	}
	return false;

    if (ctx)
        CssmError::throwMe(CSSM_ERRCODE_INTERNAL_ERROR);	// won't support re-definition
    switch (context.algorithm()) {
        case CSSM_ALGID_ROTTY_ROT_16:
            ctx = new SDContext(16);
            return true;
        case CSSM_ALGID_ROTTY_ROT_37:
            ctx = new SDContext(37);
            return true;
    }
#endif
}
