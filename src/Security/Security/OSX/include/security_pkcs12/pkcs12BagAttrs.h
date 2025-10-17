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
/*
 * pkcs12BagAttrs.h : internal representation of P12 SafeBag 
 *                    attribute, OTHER THAN friendlyName and localKeyId.
 *					  This corresponds to a SecPkcs12AttrsRef at the
 *					  public API layer.
 */
 
#ifndef	_PKCS12_BAG_ATTRS_H_
#define _PKCS12_BAG_ATTRS_H_

#include <Security/keyTemplates.h>		// for NSS_Attribute
#include <security_asn1/SecNssCoder.h>
#include <CoreFoundation/CoreFoundation.h>

class P12BagAttrs {
public:
	/* 
	 * Empty constructor used by P12SafeBag during decoding.
	 * Indivudual attrs not understood by P12SafeBag get added 
	 * via addAttr().
	 */
	P12BagAttrs(
		SecNssCoder &coder) 
		: mAttrs(NULL),
		  mCoder(coder) { }
	
	/* 
	 * Copying constructor used by P12SafeBag during encoding.
	 */
	P12BagAttrs(
		const P12BagAttrs *otherAttrs,		// optional
		SecNssCoder &coder);
		
	~P12BagAttrs() { }

	/* Raw getter used just prior to encode. */
	unsigned numAttrs() const;
	NSS_Attribute *getAttr(
		unsigned			attrNum);
		
	/*
	 * Add an attr during decoding. Only "generic" attrs, other
	 * than friendlyName and localKeyId, are added here. 
	 */
	void addAttr(
		const NSS_Attribute &attr);
	
	/*
	 * Add an attr pre-encode, from SecPkcs12Add*() or 
	 * SecPkcs12AttrsAddAttr().
	 */
	void addAttr(
		const CFDataRef		attrOid,
		const CFArrayRef	attrValues);
		
	/* 
	 * getter, public API version
	 */
	void getAttr(
		unsigned			attrNum,
		CFDataRef			*attrOid,		// RETURNED
		CFArrayRef			*attrValues);	// RETURNED
					
private:
	NSS_Attribute *reallocAttrs(
		size_t numNewAttrs);
		
	void copyAttr(
		const NSS_Attribute &src,
		NSS_Attribute &dst);
		
	/*
	 * Stored in NSS form for easy encode
	 */
	NSS_Attribute		**mAttrs;
	SecNssCoder			&mCoder;
};

/* 
 * In the most common usage, a P12BagAttrs's SecNssCoder is associated 
 * with the owning P12Coder's mCoder. In the case of a "standalone"
 * P12BagAttrs's created by the app via SecPkcs12AttrsCreate(),
 * this subclass is used, proving the P12BadAttr's SecNssCoder
 * directly.
 */
class P12BagAttrsStandAlone : public P12BagAttrs
{
public:
	P12BagAttrsStandAlone() 
		: P12BagAttrs(mPrivCoder)
			{ }

	~P12BagAttrsStandAlone() { }
	
private:
	SecNssCoder			mPrivCoder;
};

#endif	/* _PKCS12_BAG_ATTRS_H_ */

