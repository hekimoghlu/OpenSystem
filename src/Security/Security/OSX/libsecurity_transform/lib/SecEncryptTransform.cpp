/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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
#include "SecEncryptTransform.h"
#include "SecTransformInternal.h"
#include "EncryptTransform.h"
#include <utilities/SecCFRelease.h>

/* --------------------------------------------------------------------------
 Create the declared CFStringRefs
 -------------------------------------------------------------------------- */

const CFStringRef __nonnull kSecPaddingNoneKey = CFSTR("SecPaddingNoneKey");
const CFStringRef __nonnull kSecPaddingPKCS1Key = CFSTR("SecPaddingPKCS1Key");
const CFStringRef __nonnull kSecPaddingPKCS5Key = CFSTR("SecPaddingPKCS5Key");
const CFStringRef __nonnull kSecPaddingPKCS7Key = CFSTR("SecPaddingPKCS7Key");
const CFStringRef __nonnull kSecPaddingOAEPKey = CFSTR("OAEPPadding");
const CFStringRef __nonnull kSecOAEPMGF1DigestAlgorithmAttributeName = CFSTR("OAEPMGF1DigestAlgo");

const CFStringRef __nonnull kSecModeNoneKey = CFSTR("SecModeNoneKey");
const CFStringRef __nonnull kSecModeECBKey = CFSTR("SecModeECBKey");
const CFStringRef __nonnull kSecModeCBCKey = CFSTR("SecModeCBCKey");
const CFStringRef __nonnull kSecModeCFBKey = CFSTR("SecModeCFBKey");
const CFStringRef __nonnull kSecModeOFBKey = CFSTR("SecModeOFBKey");

const CFStringRef __nonnull kSecOAEPMessageLengthAttributeName = CFSTR("OAEPMessageLength");
const CFStringRef __nonnull kSecOAEPEncodingParametersAttributeName = CFSTR("OAEPEncodingParameters");

const CFStringRef __nonnull kSecEncryptKey = CFSTR("SecEncryptKey");
const CFStringRef __nonnull kSecPaddingKey = CFSTR("SecPaddingKey");
const CFStringRef __nonnull kSecIVKey = CFSTR("SecIVKey");
const CFStringRef __nonnull kSecEncryptionMode = CFSTR("SecEncryptionMode");


SecTransformRef SecEncryptTransformCreate(SecKeyRef keyRef, CFErrorRef* error)
{
	SecTransformRef etRef = EncryptTransform::Make();
	EncryptTransform* et = (EncryptTransform*) CoreFoundationHolder::ObjectFromCFType(etRef);
	if (et->InitializeObject(keyRef, error))
	{
		return etRef;
	}
	else
	{
		
		CFReleaseNull(etRef);
		return NULL;
	}
}


CFTypeID SecEncryptTransformGetTypeID()
{
	return Transform::GetCFTypeID();
}

SecTransformRef SecDecryptTransformCreate(SecKeyRef keyRef, CFErrorRef* error)
{
	
	SecTransformRef dtRef = DecryptTransform::Make();
	DecryptTransform* dt = (DecryptTransform*) CoreFoundationHolder::ObjectFromCFType(dtRef);
	if (dt->InitializeObject(keyRef, error))
	{
		return dtRef;
	}
	else
	{
		CFReleaseNull(dtRef);
		return NULL;
	}
}

CFTypeID SecDecryptTransformGetTypeID()
{
	return Transform::GetCFTypeID();
}
