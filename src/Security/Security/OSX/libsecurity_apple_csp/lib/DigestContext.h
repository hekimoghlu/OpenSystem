/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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
// DigestContext.h 
//

#ifndef	_DIGEST_CONTEXT_H_
#define _DIGEST_CONTEXT_H_

#include <security_cdsa_utilities/digestobject.h>
#include "AppleCSPContext.h"

/*
 * This is just a shim to give AppleCSPContext functionality to a 
 * DigestObject subclass (a reference to which is passed to our constructor).
 */
class DigestContext : public AppleCSPContext  {
public:
	DigestContext(
		AppleCSPSession &session,
		DigestObject &digest) : 
			AppleCSPContext(session), mDigest(digest) { }
	~DigestContext() { delete &mDigest; }
	
	void init(const Context &context, bool);
	void update(const CssmData &data);
	void final(CssmData &data);
	CSPFullPluginSession::CSPContext *clone(Allocator &);	// clone internal state
	size_t outputSize(bool, size_t);

private:
	DigestObject	&mDigest;
};

#endif	/* _CRYPTKIT_DIGEST_CONTEXT_H_ */
