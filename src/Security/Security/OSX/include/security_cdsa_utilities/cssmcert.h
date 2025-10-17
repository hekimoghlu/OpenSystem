/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 15, 2024.
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
// cssmcert - CSSM layer certificate (CL) related objects.
//
#ifndef _H_CSSMCERT
#define _H_CSSMCERT

#include <security_cdsa_utilities/cssmalloc.h>
#include <security_cdsa_utilities/cssmdata.h>


namespace Security {


//
// A CSSM_FIELD, essentially an OID/Data pair.
//
class CssmField : public PodWrapper<CssmField, CSSM_FIELD> {
public:
    CssmField() { }
    CssmField(const CSSM_OID &oid, const CSSM_DATA &value)
    { FieldOid = oid; FieldValue = value; }
    
    CssmField(const CSSM_OID &oid)
    { FieldOid = oid; FieldValue = CssmData(); }

public:
    CssmOid &oid()					{ return CssmOid::overlay(FieldOid); }
    CssmOid &value()				{ return CssmOid::overlay(FieldValue); }
    const CssmOid &oid() const		{ return CssmOid::overlay(FieldOid); }
    const CssmOid &value() const	{ return CssmOid::overlay(FieldValue); }
    
    bool isComplex() const
    { return value().length() == CSSM_FIELDVALUE_COMPLEX_DATA_TYPE; }
};


//
// An encoded certificate
//
class EncodedCertificate : public PodWrapper<EncodedCertificate, CSSM_ENCODED_CERT> {
public:
	EncodedCertificate(CSSM_CERT_TYPE type = CSSM_CERT_UNKNOWN,
		CSSM_CERT_ENCODING enc = CSSM_CERT_ENCODING_UNKNOWN,
		const CSSM_DATA *data = NULL);
	
	CSSM_CERT_TYPE type() const		{ return CertType; }
	CSSM_CERT_ENCODING encoding() const { return CertEncoding; }
	const CssmData &blob() const	{ return CssmData::overlay(CertBlob); }
	
	// CssmDataoid features
	void *data() const				{ return blob().data(); }
	size_t length() const			{ return blob().length(); }
};


//
// CertGroups - groups of certificates in a bewildering variety of forms
//
class CertGroup : public PodWrapper<CertGroup, CSSM_CERTGROUP> {
public:
    CertGroup() { }
    CertGroup(CSSM_CERT_TYPE ctype, CSSM_CERT_ENCODING encoding, CSSM_CERTGROUP_TYPE type);
    
public:
    CSSM_CERT_TYPE certType() const		{ return CertType; }
    CSSM_CERT_ENCODING encoding() const	{ return CertEncoding; }
    CSSM_CERTGROUP_TYPE type() const	{ return CertGroupType; }
    uint32 count() const				{ return NumCerts; }
    uint32 &count()						{ return NumCerts; }
    
public:
	// CSSM_CERTGROUP_DATA version
    CssmData * &blobCerts()
	{ assert(type() == CSSM_CERTGROUP_DATA); return CssmData::overlayVar(GroupList.CertList); }
    CssmData *blobCerts() const
	{ assert(type() == CSSM_CERTGROUP_DATA); return CssmData::overlay(GroupList.CertList); }
	
	// CSSM_CERTGROUP_ENCODED_CERT version
    EncodedCertificate * &encodedCerts()
		{ return EncodedCertificate::overlayVar(GroupList.EncodedCertList); }
    EncodedCertificate *encodedCerts() const
		{ return EncodedCertificate::overlay(GroupList.EncodedCertList); }
	
public:
	// free all memory in this group with the given allocator
	void destroy(Allocator &allocator);
};


//
// Walkers
//
namespace DataWalkers {




}	// end namespace DataWalkers
}	// end namespace Security

#endif //_H_CSSMCERT
