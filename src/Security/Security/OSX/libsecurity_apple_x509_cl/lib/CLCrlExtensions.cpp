/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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
 * CLCrlExtensions.cpp - CRL extensions support.
 */
 
#include "DecodedCrl.h"
#include "CLCrlExtensions.h"
#include "CLCertExtensions.h"
#include "clNssUtils.h"
#include "clNameUtils.h"
#include "CLFieldsCommon.h"
#include <security_utilities/utilities.h>
#include <Security/oidscert.h>
#include <Security/cssmerr.h>
#include <Security/x509defs.h>
#include <Security/certextensions.h>

#include <Security/SecAsn1Templates.h>

/***
 *** get/set/free functions called out from CrlFields.cpp
 ***/
/***
 *** CrlNumber , DeltaCRL
 *** CDSA format 	CE_CrlNumber (a uint32)
 *** NSS format 	CSSM_DATA, length 4
 *** OID 			CSSMOID_CrlNumber, CSSMOID_DeltaCrlIndicator
 ***/
 
/* set function for both */
void setFieldCrlNumber(		
	DecodedItem	&crl, 
	const CssmData &fieldValue) 
{
	CSSM_X509_EXTENSION_PTR cssmExt = verifySetFreeExtension(fieldValue, 
		false);
	CE_CrlNumber *cdsaObj = (CE_CrlNumber *)cssmExt->value.parsedValue;
	
	/* CSSM_DATA and its contents in crl.coder's memory */
	ArenaAllocator alloc(crl.coder());
	CSSM_DATA_PTR nssVal = (CSSM_DATA_PTR)alloc.malloc(sizeof(CSSM_DATA));
	clIntToData(*cdsaObj, *nssVal, alloc);
	
	/* add to mExtensions */
	crl.addExtension(nssVal, cssmExt->extnId, cssmExt->critical, false,
		kSecAsn1IntegerTemplate); 
}

static
bool getFieldCrlCommon(
	DecodedItem		 	&crl,
	const CSSM_OID		&fieldId,		// identifies extension we seek
	unsigned			index,			// which occurrence (0 = first)
	uint32				&numFields,		// RETURNED
	CssmOwnedData		&fieldValue) 
{
	const DecodedExten *decodedExt;
	CSSM_DATA *nssObj;
	CE_CrlNumber *cdsaObj;
	bool brtn;
	
	brtn = crl.GetExtenTop<CSSM_DATA, CE_CrlNumber>(
		index,
		numFields,
		fieldValue.allocator,
		fieldId,
		nssObj,
		cdsaObj,
		decodedExt);
	if(!brtn) {
		return false;
	}
	*cdsaObj = clDataToInt(*nssObj, CSSMERR_CL_INVALID_CRL_POINTER);
	
	/* pass back to caller */
	getFieldExtenCommon(cdsaObj, *decodedExt, fieldValue);
	return true;
}

bool getFieldCrlNumber(
	DecodedItem		 	&crl,
	unsigned			index,			// which occurrence (0 = first)
	uint32				&numFields,		// RETURNED
	CssmOwnedData		&fieldValue) 
{
	return getFieldCrlCommon(crl, CSSMOID_CrlNumber, index, numFields, 
		fieldValue);
}

bool getFieldDeltaCrl(
	DecodedItem		 	&crl,
	unsigned			index,			// which occurrence (0 = first)
	uint32				&numFields,		// RETURNED
	CssmOwnedData		&fieldValue) 
{
	return getFieldCrlCommon(crl, CSSMOID_DeltaCrlIndicator, index, 
		numFields, fieldValue);
}

void freeFieldIssuingDistPoint (
	CssmOwnedData		&fieldValue)
{
	CSSM_X509_EXTENSION_PTR cssmExt = verifySetFreeExtension(fieldValue, false);
	Allocator &alloc = fieldValue.allocator;
	CE_IssuingDistributionPoint *cdsaObj = 
			(CE_IssuingDistributionPoint *)cssmExt->value.parsedValue;
	CL_freeCssmIssuingDistPoint(cdsaObj, alloc);
	freeFieldExtenCommon(cssmExt, alloc);		// frees extnId, parsedValue, BERvalue
}

void freeFieldCrlDistributionPoints (
	CssmOwnedData		&fieldValue)
{
	CSSM_X509_EXTENSION_PTR cssmExt = verifySetFreeExtension(fieldValue, false);
	Allocator &alloc = fieldValue.allocator;
	CE_CRLDistPointsSyntax *cdsaObj = 
			(CE_CRLDistPointsSyntax *)cssmExt->value.parsedValue;
	CL_freeCssmDistPoints(cdsaObj, alloc);
	freeFieldExtenCommon(cssmExt, alloc);		// frees extnId, parsedValue, BERvalue
}

/* HoldInstructionCode - CSSM_OID */
/* InvalidityDate - CSSM_DATA */
void freeFieldOidOrData (
	CssmOwnedData		&fieldValue)
{
	CSSM_X509_EXTENSION_PTR cssmExt = verifySetFreeExtension(fieldValue, false);
	Allocator &alloc = fieldValue.allocator;
	CSSM_DATA *cdsaObj = 
			(CSSM_DATA *)cssmExt->value.parsedValue;
	if(cdsaObj) {
		alloc.free(cdsaObj->Data);
	}
	freeFieldExtenCommon(cssmExt, alloc);		// frees extnId, parsedValue, BERvalue
}

