/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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
 *  SecIdentity.c - CoreFoundation based object containing a
 *  private key, certificate tuple.
 */


#include <Security/SecIdentity.h>

#include <CoreFoundation/CFRuntime.h>
#include <CoreFoundation/CFString.h>
#include <Security/SecCertificate.h>
#include <Security/SecKey.h>
#include <pthread.h>
#include <Security/SecIdentityPriv.h>
#include <Security/SecInternal.h>
#include <utilities/SecCFWrappers.h>

struct __SecIdentity {
    CFRuntimeBase		_base;
	SecCertificateRef	_certificate;
	SecKeyRef			_privateKey;
};

CFGiblisWithHashFor(SecIdentity)

/* Static functions. */
static CFStringRef SecIdentityCopyFormatDescription(CFTypeRef cf, CFDictionaryRef formatOptions) {
    SecIdentityRef identity = (SecIdentityRef)cf;
    return CFStringCreateWithFormat(kCFAllocatorDefault, NULL,
        CFSTR("<SecIdentityRef: %p>"), identity);
}

static void SecIdentityDestroy(CFTypeRef cf) {
    SecIdentityRef identity = (SecIdentityRef)cf;
	CFReleaseNull(identity->_certificate);
	CFReleaseNull(identity->_privateKey);
}

static Boolean SecIdentityCompare(CFTypeRef cf1, CFTypeRef cf2) {
    SecIdentityRef identity1 = (SecIdentityRef)cf1;
    SecIdentityRef identity2 = (SecIdentityRef)cf2;
    if (identity1 == identity2)
        return true;
    if (!identity2)
        return false;
    return CFEqual(identity1->_certificate, identity2->_certificate) &&
		CFEqual(identity1->_privateKey, identity2->_privateKey);
}

/* Hash of identity is hash of certificate plus hash of key. */
static CFHashCode SecIdentityHash(CFTypeRef cf) {
    SecIdentityRef identity = (SecIdentityRef)cf;
	return CFHash(identity->_certificate) + CFHash(identity->_privateKey);
}

OSStatus SecIdentityCopyCertificate(SecIdentityRef identity,
	SecCertificateRef *certificateRef) {
	*certificateRef = identity->_certificate;
	CFRetain(*certificateRef);
	return 0;
}

OSStatus SecIdentityCopyPrivateKey(SecIdentityRef identity,
	SecKeyRef *privateKeyRef) {
	*privateKeyRef = identity->_privateKey;
	CFRetain(*privateKeyRef);
	return 0;
}

SecIdentityRef SecIdentityCreate(CFAllocatorRef allocator,
	SecCertificateRef certificate, SecKeyRef privateKey) {
    if (!certificate || CFGetTypeID(certificate) != SecCertificateGetTypeID() ||
        !privateKey || CFGetTypeID(privateKey) != SecKeyGetTypeID()) {
        return NULL;
    }
    CFIndex size = sizeof(struct __SecIdentity);
    SecIdentityRef result = (SecIdentityRef)_CFRuntimeCreateInstance(
		allocator, SecIdentityGetTypeID(), size - sizeof(CFRuntimeBase), 0);
	if (result) {
		CFRetain(certificate);
		CFRetain(privateKey);
		result->_certificate = certificate;
		result->_privateKey = privateKey;
    }
    return result;
}

