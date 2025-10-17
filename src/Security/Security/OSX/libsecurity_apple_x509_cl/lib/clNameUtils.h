/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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
 * clNameUtils.h - support for Name, GeneralizedName, all sorts of names
 */
 
#ifndef	_CL_NAME_UTILS_H_
#define _CL_NAME_UTILS_H_

#include <Security/cssmtype.h>
#include <security_utilities/alloc.h>
#include <Security/x509defs.h>
#include <Security/certextensions.h>
#include <Security/X509Templates.h>
#include <security_asn1/SecNssCoder.h>

void CL_nssAtvToCssm(
	const NSS_ATV				&nssObj,
	CSSM_X509_TYPE_VALUE_PAIR	&cssmObj,
	Allocator				&alloc
	#if !NSS_TAGGED_ITEMS
	, SecNssCoder					&coder
	#endif
	);
void CL_nssRdnToCssm(
	const NSS_RDN				&nssObj,
	CSSM_X509_RDN				&cssmObj,
	Allocator				&alloc,
	SecNssCoder					&coder);
void CL_nssNameToCssm(
	const NSS_Name				&nssObj,
	CSSM_X509_NAME				&cssmObj,
	Allocator				&alloc);

void CL_cssmAtvToNss(
	const CSSM_X509_TYPE_VALUE_PAIR	&cssmObj,
	NSS_ATV							&nssObj,
	SecNssCoder						&coder);
void CL_cssmRdnToNss(
	const CSSM_X509_RDN			&cssmObj,
	NSS_RDN						&nssObj,
	SecNssCoder					&coder);
void CL_cssmNameToNss(
	const CSSM_X509_NAME		&cssmObj,
	NSS_Name					&nssObj,
	SecNssCoder					&coder);

void CL_normalizeString(
	char 						*strPtr,
	int 						&strLen);		// IN/OUT
void CL_normalizeX509NameNSS(
	NSS_Name 					&nssName,
	SecNssCoder 				&coder);

void CL_nssGeneralNameToCssm(
	NSS_GeneralName &nssObj,
	CE_GeneralName &cdsaObj,
	SecNssCoder &coder,				// for temp decoding
	Allocator &alloc);			// destination 

void CL_nssGeneralNamesToCssm(
	const NSS_GeneralNames &nssObj,
	CE_GeneralNames &cdsaObj,
	SecNssCoder &coder,				// for temp decoding
	Allocator &alloc);			// destination 
void CL_cssmGeneralNameToNss(
	CE_GeneralName &cdsaObj,
	NSS_GeneralName &nssObj,		// actually an NSSTaggedItem
	SecNssCoder &coder);			// for temp decoding
void CL_cssmGeneralNamesToNss(
	const CE_GeneralNames &cdsaObj,
	NSS_GeneralNames &nssObj,
	SecNssCoder &coder);
	
void clCopyOtherName(
	const CE_OtherName 			&src,
	CE_OtherName 				&dst,
	Allocator					&alloc);

void CL_freeAuthorityKeyId(
	CE_AuthorityKeyID			&cdsaObj,
	Allocator					&alloc);
void CL_freeCssmGeneralName(
	CE_GeneralName				&genName,
	Allocator					&alloc);
void CL_freeCssmGeneralNames(
	CE_GeneralNames				*cdsaObj,
	Allocator					&alloc);
void CL_freeCssmDistPointName(
	CE_DistributionPointName	*cssmDpn,
	Allocator					&alloc);
void CL_freeCssmDistPoints(
	CE_CRLDistPointsSyntax		*cssmDps,
	Allocator					&alloc);
void CL_freeX509Name(
	CSSM_X509_NAME_PTR			x509Name,
	Allocator					&alloc);
void CL_freeX509Rdn(
	CSSM_X509_RDN_PTR			rdn,
	Allocator					&alloc);
void CL_freeOtherName(
	CE_OtherName				*cssmOther,
	Allocator					&alloc);
void CL_freeCssmIssuingDistPoint(
	CE_IssuingDistributionPoint	*cssmIdp,
	Allocator					&alloc);


#endif	/* _CL_NAME_UTILS_H_ */
