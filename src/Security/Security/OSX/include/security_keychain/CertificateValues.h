/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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
// CertificateValues.h - Objects in a Certificate
//
#ifndef _SECURITY_CERTIFICATEVALUES_H_
#define _SECURITY_CERTIFICATEVALUES_H_

#include <Security/SecBase.h>


namespace Security
{

namespace KeychainCore
{

class CertificateValues
{
	NOCOPY(CertificateValues)

public:

	CertificateValues(SecCertificateRef certificateRef);
	virtual ~CertificateValues() _NOEXCEPT;

	static CFStringRef remapLabelToKey(CFStringRef label);
	CFArrayRef copyPropertyValues(CFErrorRef *error);
	CFDictionaryRef copyFieldValues(CFArrayRef keys, CFErrorRef *error);
	CFDataRef copySerialNumber(CFErrorRef *error);
	CFDataRef copyNormalizedIssuerContent(CFErrorRef *error);
	CFDataRef copyNormalizedSubjectContent(CFErrorRef *error);
	CFDataRef copyIssuerSequence(CFErrorRef *error);
	CFDataRef copySubjectSequence(CFErrorRef *error);
	CFStringRef copyIssuerSummary(CFErrorRef *error);
	CFStringRef copySubjectSummary(CFErrorRef *error);
	CFDictionaryRef copyAttributeDictionary(CFErrorRef *error);
	bool isValid(CFAbsoluteTime verifyTime, CFErrorRef *error);
	CFAbsoluteTime notValidBefore(CFErrorRef *error);
	CFAbsoluteTime notValidAfter(CFErrorRef *error);

private:

	SecCertificateRef copySecCertificateRef(CFErrorRef *error);

	SecCertificateRef mCertificateRef;
	CFDataRef mCertificateData;
	CFArrayRef mCertificateProperties;
	static CFDictionaryRef mOIDRemap;
};


} // end namespace KeychainCore

} // end namespace Security

#endif // !_SECURITY_CERTIFICATEVALUES_H_
