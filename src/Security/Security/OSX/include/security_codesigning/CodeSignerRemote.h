/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
// CodeSignerRemote - SecCodeSignerRemote API objects
//
#ifndef _H_CODESIGNERREMOTE
#define _H_CODESIGNERREMOTE

#include "cs.h"
#include "StaticCode.h"
#include "cdbuilder.h"
#include <Security/SecIdentity.h>
#include <security_utilities/utilities.h>
#include "SecCodeSigner.h"
#include "CodeSigner.h"

namespace Security {
namespace CodeSigning {


//
// SecCodeSigner is responsible for signing code objects
//
class SecCodeSignerRemote : public SecCodeSigner {
	NOCOPY(SecCodeSignerRemote)

public:
	SECCFFUNCTIONS(SecCodeSignerRemote, SecCodeSignerRemoteRef, errSecCSInvalidObjectRef, gCFObjects().CodeSignerRemote)

	SecCodeSignerRemote(SecCSFlags flags, CFArrayRef certificateChain);

	virtual bool valid() const;
	void sign(SecStaticCode *code, SecCSFlags flags, SecCodeRemoteSignHandler handler);

	virtual ~SecCodeSignerRemote() _NOEXCEPT;

public:
	SecCodeRemoteSignHandler mSignHandler;
	CFRef<CFArrayRef> mCertificateChain;
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_CODESIGNERREMOTE
