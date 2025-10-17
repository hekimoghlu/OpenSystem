/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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
 * CLCertExtensions.h - extern declarations of get/set/free functions implemented in
 *                    CertExtensions,cpp and used only in CertFields.cpp.
 */

#ifndef	_CL_CERT_EXTENSIONS_H_
#define _CL_CERT_EXTENSIONS_H_

#include "DecodedCert.h"
#include "CLFieldsCommon.h"

#ifdef	__cplusplus
extern "C" {
#endif

/*
 * Functions to map OID --> {get,set,free}field
 */
getItemFieldFcn getFieldKeyUsage, getFieldBasicConstraints, 
	getFieldExtKeyUsage,
	getFieldSubjectKeyId, getFieldAuthorityKeyId, getFieldSubjAltName,
	getFieldIssuerAltName,
	getFieldCertPolicies, getFieldNetscapeCertType, getFieldCrlDistPoints,
	getFieldAuthInfoAccess, getFieldSubjInfoAccess, getFieldUnknownExt,
	getFieldQualCertStatements,
	getFieldNameConstraints, getFieldPolicyMappings, getFieldPolicyConstraints,
	getFieldInhibitAnyPolicy;
setItemFieldFcn setFieldKeyUsage, setFieldBasicConstraints, 
	setFieldExtKeyUsage,
	setFieldSubjectKeyId, setFieldAuthorityKeyId, setFieldSubjIssuerAltName,
	setFieldCertPolicies, setFieldNetscapeCertType, setFieldCrlDistPoints,
	setFieldAuthInfoAccess, setFieldUnknownExt, setFieldQualCertStatements,
	setFieldNameConstraints, setFieldPolicyMappings, setFieldPolicyConstraints,
	setFieldInhibitAnyPolicy;
freeFieldFcn freeFieldExtKeyUsage, freeFieldSubjectKeyId,
	freeFieldAuthorityKeyId, freeFieldSubjIssuerAltName, 
	freeFieldCertPolicies, 
	freeFieldCrlDistPoints, freeFieldInfoAccess, freeFieldUnknownExt,
	freeFieldQualCertStatements,
	freeFieldNameConstraints, freeFieldPolicyMappings, freeFieldPolicyConstraints;
	
#ifdef	__cplusplus
}
#endif

#endif	/* _CERT_EXTENSIONS_H_*/
