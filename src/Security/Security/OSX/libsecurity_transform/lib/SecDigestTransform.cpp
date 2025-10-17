/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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

#include "SecDigestTransform.h"
#include "SecCFRelease.h"
#include "Digest.h"



const CFStringRef kSecDigestMD2 = CFSTR("MD2 Digest"),
				  kSecDigestMD4 = CFSTR("MD4 Digest"),
				  kSecDigestMD5 = CFSTR("MD5 Digest"),
				  kSecDigestSHA1 = CFSTR("SHA1 Digest"),
				  kSecDigestSHA2 = CFSTR("SHA2 Digest Family"),
				  kSecDigestHMACMD5 = CFSTR("HMAC-MD5"),
				  kSecDigestHMACSHA1 = CFSTR("HMAC-SHA1"),
				  kSecDigestHMACSHA2 = CFSTR("HMAC-SHA2 Digest Family");

const CFStringRef kSecDigestTypeAttribute = CFSTR("Digest Type"),
				  kSecDigestLengthAttribute = CFSTR("Digest Length"),
				  kSecDigestHMACKeyAttribute = CFSTR("HMAC Key");

SecTransformRef SecDigestTransformCreate(CFTypeRef digestType,
										 CFIndex digestLength,
										 CFErrorRef* error
										 )
{
	SecTransformRef tr = DigestTransform::Make();
	DigestTransform* dt = (DigestTransform*) CoreFoundationHolder::ObjectFromCFType(tr);
	
	CFErrorRef result = dt->Setup(digestType, digestLength);
	if (result != NULL)
	{
		// an error occurred
		CFReleaseNull(tr);
		
		if (error)
		{
            CFRetainSafe(result);
			*error = result;
		}
		
		return NULL;
	}
	
	return tr;
}



CFTypeID SecDigestTransformGetTypeID()
{
	return DigestTransform::GetCFTypeID();
}
