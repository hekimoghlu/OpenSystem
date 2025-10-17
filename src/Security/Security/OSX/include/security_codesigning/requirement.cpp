/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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
// requirement - Code Requirement Blob description
//
#include "requirement.h"
#include "reqinterp.h"
#include "codesigning_dtrace.h"
#include <security_utilities/errors.h>
#include <security_utilities/unix++.h>
#include <security_utilities/logging.h>
#include <security_utilities/cfutilities.h>
#include <security_utilities/hashing.h>
#include "LWCRHelper.h"

#ifdef DEBUGDUMP
#include "reqdumper.h"
#endif

namespace Security {
namespace CodeSigning {


//
// Canonical names for requirement types
//
const char *const Requirement::typeNames[] = {
	"invalid",
	"host",
	"guest",
	"designated",
	"library",
	"plugin",
};


//
// validate a requirement against a code context
//
void Requirement::validate(const Requirement::Context &ctx, OSStatus failure /* = errSecCSReqFailed */) const
{
	if (!this->validates(ctx, failure))
		MacOSError::throwMe(failure);
}

bool Requirement::validates(const Requirement::Context &ctx, OSStatus failure /* = errSecCSReqFailed */) const
{
	CODESIGN_EVAL_REQINT_START((void*)this, (int)this->length());
	switch (kind()) {
	case exprForm:
		if (Requirement::Interpreter(this, &ctx).evaluate()) {
			CODESIGN_EVAL_REQINT_END(this, 0);
			return true;
		} else {
			CODESIGN_EVAL_REQINT_END(this, failure);
			return false;
		}
#if !TARGET_OS_SIMULATOR
	case lwcrForm: {
		CFRef<CFDataRef> lwcr = createlwcrFormData();
		if (evaluateLightweightCodeRequirement(ctx, lwcr)) {
			CODESIGN_EVAL_REQINT_END(this, 0);
			return true;
		} else {
			CODESIGN_EVAL_REQINT_END(this, failure);
			return false;
		}
	}
#endif
	default:
		CODESIGN_EVAL_REQINT_END(this, errSecCSReqUnsupported);
		MacOSError::throwMe(errSecCSReqUnsupported);
	}
}

CFDataRef Requirement::createlwcrFormData() const
{
	if (kind() == lwcrForm) {
		Requirement::Reader reader(this);
		const UInt8* data = NULL;
		size_t length = 0;
		reader.getData(data, length);
		return CFDataCreate(NULL, data, length);
	} else {
		MacOSError::throwMe(errSecCSReqInvalid);
	}
}


//
// Retrieve one certificate from the cert chain.
// Positive and negative indices can be used:
//    [ leaf, intermed-1, ..., intermed-n, anchor ]
//        0       1       ...     -2         -1
// Returns NULL if unavailable for any reason.
//	
SecCertificateRef Requirement::Context::cert(int ix) const
{
	if (certs) {
		if (ix < 0)
			ix += certCount();
		if (ix >= CFArrayGetCount(certs))
		    return NULL;
		if (CFTypeRef element = CFArrayGetValueAtIndex(certs, ix))
			return SecCertificateRef(element);
	}
	return NULL;
}

unsigned int Requirement::Context::certCount() const
{
	if (certs)
		return (unsigned int)CFArrayGetCount(certs);
	else
		return 0;
}


//
// Produce the hash of a fake Apple root (only if compiled for internal testing)
//
#if defined(TEST_APPLE_ANCHOR)

const char Requirement::testAppleAnchorEnv[] = "TEST_APPLE_ANCHOR";

const SHA1::Digest &Requirement::testAppleAnchorHash()
{
	static bool tried = false;
	static SHA1::Digest testHash;
	if (!tried) {
		// see if we have one configured
		if (const char *path = getenv(testAppleAnchorEnv))
			try {
				UnixPlusPlus::FileDesc fd(path);
				char buffer[2048];		// arbitrary limit
				size_t size = fd.read(buffer, sizeof(buffer));
				SHA1 hash;
				hash(buffer, size);
				hash.finish(testHash);
				Syslog::alert("ACCEPTING TEST AUTHORITY %s FOR APPLE CODE IDENTITY", path);
			} catch (...) { }
		tried = true;
	}
	return testHash;		// will be zeroes (no match) if not configured
}

#endif //TEST_APPLE_ANCHOR

//
// Debug dump support
//
#if TARGET_OS_OSX
#ifdef DEBUGDUMP

void Requirement::dump() const
{
	Debug::dump("%s\n", Dumper::dump(this).c_str());
}

#endif //DEBUGDUMP
#endif


}	// CodeSigning
}	// Security
