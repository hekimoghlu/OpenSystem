/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
// xar++ - interface to XAR-format archive files
//
#include "xar++.h"
#include "notarization.h"
#include <security_utilities/cfutilities.h>
#include <Security/Security.h>


namespace Security {
namespace CodeSigning {


Xar::Xar(const char *path)
{
	mXar = 0;
	mSigCMS = 0;
	mSigClassic = 0;
	if (path)
		open(path);
}

void Xar::open(const char *path)
{
	if ((mXar = ::xar_open(path, READ)) == NULL)
	    return;

	mPath = std::string(path);
    
	xar_signature_t sig = ::xar_signature_first(mXar);
	// read signatures until we find a CMS signature
	while (sig && mSigCMS == NULL) {
		const char *type = ::xar_signature_type(sig);
		if (strcmp(type, "CMS") == 0) {
			mSigCMS = sig;
		} else if (strcmp(type, "RSA") == 0) {
			mSigClassic = sig;
		}
		sig = ::xar_signature_next(sig);
	}
}

Xar::~Xar()
{
	if (mXar)
		::xar_close(mXar);
}

static CFArrayRef copyCertChainFromSignature(xar_signature_t sig)
{
	unsigned count = xar_signature_get_x509certificate_count(sig);
	CFRef<CFMutableArrayRef> certs = makeCFMutableArray(0);
	for (unsigned ix = 0; ix < count; ix++) {
		const uint8_t *data;
		uint32_t length;
		if (xar_signature_get_x509certificate_data(sig, ix, &data, &length) == 0) {
			CFTempData cdata(data, length);
			CFRef<SecCertificateRef> cert = SecCertificateCreateWithData(NULL, cdata);
			CFArrayAppendValue(certs, cert.get());
		}
	}
	return certs.yield();
}

CFArrayRef Xar::copyCertChain()
{
	if (mSigCMS)
		return copyCertChainFromSignature(mSigCMS);
	else if (mSigClassic)
		return copyCertChainFromSignature(mSigClassic);
	return NULL;
}

void Xar::registerStapledNotarization()
{
	registerStapledTicketInPackage(mPath);
}

CFDataRef Xar::createPackageChecksum()
{
	xar_signature_t sig = NULL;

	// Always prefer a CMS signature to a class signature and return early
	// if no appropriate signature has been found.
	if (mSigCMS) {
		sig = mSigCMS;
	} else if (mSigClassic) {
		sig = mSigClassic;
	} else {
		return NULL;
	}

	// Extract the signed data from the xar, which is actually just the checksum
	// we use as an identifying hash.
	uint8_t *data = NULL;
	uint32_t length;
	if (xar_signature_copy_signed_data(sig, &data, &length, NULL, NULL, NULL) != 0) {
		secerror("Unable to extract package hash for package: %s", mPath.c_str());
		return NULL;
	}

	// xar_signature_copy_signed_data returns malloc'd data that can be used without copying
	// but must be free'd properly later.
	return makeCFDataMalloc(data, length);
}

SecCSDigestAlgorithm Xar::checksumDigestAlgorithm()
{
	int32_t error = 0;
	const char* value = NULL;
	unsigned long size = 0;

	if (mXar == NULL) {
		secerror("Evaluating checksum digest on bad xar: %s", mPath.c_str());
		return kSecCodeSignatureNoHash;
	}

	error = xar_prop_get((xar_file_t)mXar, "checksum/size", &value);
	if (error == -1) {
		secerror("Unable to extract package checksum size: %s", mPath.c_str());
		return kSecCodeSignatureNoHash;
	}

	size = strtoul(value, NULL, 10);
	switch (size) {
		case CC_SHA1_DIGEST_LENGTH:
			return kSecCodeSignatureHashSHA1;
		case CC_SHA256_DIGEST_LENGTH:
			return kSecCodeSignatureHashSHA256;
		case CC_SHA512_DIGEST_LENGTH:
			return kSecCodeSignatureHashSHA512;
		case CC_MD5_DIGEST_LENGTH:
		default:
			return kSecCodeSignatureNoHash;
	}
}

} // end namespace CodeSigning
} // end namespace Security
