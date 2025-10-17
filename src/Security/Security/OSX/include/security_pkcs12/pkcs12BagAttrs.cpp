/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
 * pkcs12BagAttrs.cpp : internal representation of P12 SafeBag 
 *                      attribute, OTHER THAN friendlyName and localKeyId.
 *					    This corresponds to a SecPkcs12AttrsRef at the
 *					    public API layer.
 */

#include "pkcs12BagAttrs.h"
#include "pkcs12Utils.h"
#include <security_asn1/nssUtils.h>
#include <security_utilities/simulatecrash_assert.h>
#include <Security/SecBase.h>
/* 
 * Copying constructor used by P12SafeBag during encoding
 */
P12BagAttrs::P12BagAttrs(
	const P12BagAttrs *otherAttrs,
	SecNssCoder &coder)
		: mAttrs(NULL), mCoder(coder)
{
	if(otherAttrs == NULL) {
		/* empty copy, done */
		return;
	}
	unsigned num = otherAttrs->numAttrs();
	reallocAttrs(num);
	for(unsigned dex=0; dex<num; dex++) {
		copyAttr(*otherAttrs->mAttrs[dex], *mAttrs[dex]);
	}
}

unsigned P12BagAttrs::numAttrs() const
{
	return nssArraySize((const void **)mAttrs);
}

NSS_Attribute *P12BagAttrs::getAttr(
	unsigned attrNum)
{
	assert(attrNum < numAttrs());
	return mAttrs[attrNum];
}
		

/*
 * Add an attr during decode.
 */
void P12BagAttrs::addAttr(
	const NSS_Attribute &attr)
{
	NSS_Attribute *newAttr = reallocAttrs(numAttrs() + 1);
	copyAttr(attr, *newAttr);
}

/*
 * Add an attr during encode.
 */
void P12BagAttrs::addAttr(
	const CFDataRef		attrOid,
	const CFArrayRef	attrValues)
{
	NSS_Attribute *newAttr = reallocAttrs(numAttrs() + 1);
	p12CfDataToCssm(attrOid, newAttr->attrType, mCoder);
	uint32 numVals = (uint32)CFArrayGetCount(attrValues);
	newAttr->attrValue = (CSSM_DATA **)p12NssNullArray(numVals, mCoder);
	for(unsigned dex=0; dex<numVals; dex++) {
		CSSM_DATA *dstVal = (CSSM_DATA *)mCoder.alloc(sizeof(CSSM_DATA));
		newAttr->attrValue[dex] = dstVal;
		CFDataRef srcVal = (CFDataRef)CFArrayGetValueAtIndex(attrValues, dex);
		assert(CFGetTypeID(srcVal) == CFDataGetTypeID());
		p12CfDataToCssm(srcVal, *dstVal, mCoder);
	}
}

/* 
 * getter, public API version
 */
void P12BagAttrs::getAttr(
	unsigned			attrNum,
	CFDataRef			*attrOid,		// RETURNED
	CFArrayRef			*attrValues)	// RETURNED
{
	if(attrNum >= numAttrs()) {
		MacOSError::throwMe(errSecParam);
	}
	NSS_Attribute *attr = mAttrs[attrNum];
	*attrOid = p12CssmDataToCf(attr->attrType);
	unsigned numVals = nssArraySize((const void **)attr->attrValue);
	if(numVals == 0) {
		/* maybe should return empty array...? */
		*attrValues = NULL;
		return;
	}
	CFMutableArrayRef vals = CFArrayCreateMutable(NULL, numVals, NULL);
	for(unsigned dex=0; dex<numVals; dex++) {
		CFDataRef val = p12CssmDataToCf(*attr->attrValue[dex]);
		CFArrayAppendValue(vals, val);
        CFRelease(val);
	}
	*attrValues = vals;
}

#pragma mark --- private methods ---

/*
 * Alloc/realloc attr array.
 * Returns ptr to new empty NSS_Attribute for insertion.
 */
NSS_Attribute *P12BagAttrs::reallocAttrs(
	size_t numNewAttrs)
{
	unsigned curSize = numAttrs();
	assert(numNewAttrs > curSize);
	NSS_Attribute **newAttrs = 
		(NSS_Attribute **)p12NssNullArray((uint32)numNewAttrs, mCoder);
	for(unsigned dex=0; dex<curSize; dex++) {
		newAttrs[dex] = mAttrs[dex];
	}
	mAttrs = newAttrs;
	
	/* allocate new NSS_Attributes */
	for(unsigned dex=curSize; dex<numNewAttrs; dex++) {
		mAttrs[dex] = mCoder.mallocn<NSS_Attribute>();
		memset(mAttrs[dex], 0, sizeof(NSS_Attribute));
	}
	return mAttrs[curSize];
}

void P12BagAttrs::copyAttr(
	const NSS_Attribute &src,
	NSS_Attribute &dst)
{
	mCoder.allocCopyItem(src.attrType, dst.attrType);
	unsigned numVals = nssArraySize((const void **)src.attrValue);
	dst.attrValue = (CSSM_DATA **)p12NssNullArray(numVals, mCoder);
	for(unsigned dex=0; dex<numVals; dex++) {
		CSSM_DATA *dstVal = mCoder.mallocn<CSSM_DATA>();
		memset(dstVal, 0, sizeof(CSSM_DATA));
		dst.attrValue[dex] = dstVal;
		mCoder.allocCopyItem(*src.attrValue[dex], *dstVal);
	}
}
