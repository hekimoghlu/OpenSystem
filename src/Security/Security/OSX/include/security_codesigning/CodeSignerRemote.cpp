/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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
// CodeRemoteSigner - SecCodeRemoteSigner API objects
//
#include "CodeSignerRemote.h"
#include "signer.h"
#include "csdatabase.h"
#include "drmaker.h"
#include "csutilities.h"
#include <security_utilities/unix++.h>
#include <security_utilities/unixchild.h>
#include <Security/SecCertificate.h>
#include <Security/SecCertificatePriv.h>
#include <Security/SecPolicy.h>
#include <Security/SecPolicyPriv.h>
#include <libDER/oids.h>
#include <vector>
#include <errno.h>

namespace Security {

namespace CodeSigning {

using namespace UnixPlusPlus;

//
// Construct a SecCodeSignerRemote
//
SecCodeSignerRemote::SecCodeSignerRemote(SecCSFlags flags, CFArrayRef certificateChain)
: SecCodeSigner(flags), mCertificateChain(NULL)
{
	// Set here vs the initializer to ensure we take a reference.
	mCertificateChain = certificateChain;
}

//
// Clean up a SecCodeSignerRemote
//
SecCodeSignerRemote::~SecCodeSignerRemote() _NOEXCEPT
{
}

bool
SecCodeSignerRemote::valid() const
{
	bool isValid = true;

	// Must have a certificate chain that is a valid array of at least one certificate.
	bool arrayExists = mCertificateChain.get() != NULL;
	bool arrayHasItems = false;
	bool arrayHasCorrectItems = true;

	if (arrayExists) {
		CFIndex count = CFArrayGetCount(mCertificateChain.get());
		arrayHasItems = count > 0;

		if (arrayHasItems) {
			for (CFIndex i = 0; i < count; i++) {
				CFTypeRef obj = CFArrayGetValueAtIndex(mCertificateChain.get(), i);
				if (SecCertificateGetTypeID() != CFGetTypeID(obj)) {
					arrayHasCorrectItems = false;
					break;
				}
			}
		}
	}

	isValid = arrayExists && arrayHasItems && arrayHasCorrectItems;
	if (!isValid) {
		secerror("Invalid remote signing operation: %p, %@", this, mCertificateChain.get());
	}
	return isValid;
}

void
SecCodeSignerRemote::sign(SecStaticCode *code, SecCSFlags flags, SecCodeRemoteSignHandler handler)
{
	// Never preserve a linker signature.
	if (code->isSigned() &&
		(flags & kSecCSSignPreserveSignature) &&
		!code->flag(kSecCodeSignatureLinkerSigned)) {
		return;
	}

	secinfo("remotesigner", "%p will start remote signature from %p with block %p", this, code, handler);

	code->setValidationFlags(flags);
	Signer operation(*this, code);

	if (!valid()) {
		secerror("Invalid signing operation, bailing.");
		MacOSError::throwMe(errSecCSInvalidObjectRef);
	}
	secinfo("remotesigner", "%p will sign %p (flags 0x%x) with certs: %@", this, code, flags, mCertificateChain.get());
	operation.setupRemoteSigning(mCertificateChain, handler);
	operation.sign(flags);
	code->resetValidity();
}


} // end namespace CodeSigning
} // end namespace Security
