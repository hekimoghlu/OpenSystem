/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 28, 2024.
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
// PODWrapper for CssmKey and related types
//
#ifndef _H_CSSMKEY
#define _H_CSSMKEY

#include <security_utilities/utilities.h>
#include <security_cdsa_utilities/cssmpods.h>
#include <security_cdsa_utilities/cssmerrors.h>
#include <Security/cssm.h>


namespace Security {


//
// User-friendlier CSSM_KEY objects
//
class CssmKey : public PodWrapper<CssmKey, CSSM_KEY> {
public:
    CssmKey() { clearPod(); KeyHeader.HeaderVersion = CSSM_KEYHEADER_VERSION; }
	// all of the following constructors take over ownership of the key data
    CssmKey(const CSSM_KEY &key);
	CssmKey(const CSSM_DATA &keyData);
    CssmKey(uint32 length, void *data);

public:
    class Header : public PodWrapper<Header, CSSM_KEYHEADER> {
    public:
		// access to components of the key header
		CSSM_KEYBLOB_TYPE blobType() const { return BlobType; }
		void blobType(CSSM_KEYBLOB_TYPE blobType) { BlobType = blobType; }
		
		CSSM_KEYBLOB_FORMAT blobFormat() const { return Format; }
		void blobFormat(CSSM_KEYBLOB_FORMAT blobFormat) { Format = blobFormat; }
		
		CSSM_KEYCLASS keyClass() const { return KeyClass; }
		void keyClass(CSSM_KEYCLASS keyClass) { KeyClass = keyClass; }
		
		CSSM_KEY_TYPE algorithm() const { return AlgorithmId; }
		void algorithm(CSSM_KEY_TYPE algorithm) { AlgorithmId = algorithm; }
		
		CSSM_KEY_TYPE wrapAlgorithm() const { return WrapAlgorithmId; }
		void wrapAlgorithm(CSSM_KEY_TYPE wrapAlgorithm) { WrapAlgorithmId = wrapAlgorithm; }
        
        CSSM_ENCRYPT_MODE wrapMode() const { return WrapMode; }
        void wrapMode(CSSM_ENCRYPT_MODE mode) { WrapMode = mode; }
		
		bool isWrapped() const { return WrapAlgorithmId != CSSM_ALGID_NONE; }

		const Guid &cspGuid() const { return Guid::overlay(CspId); }
		void cspGuid(const Guid &guid) { Guid::overlay(CspId) = guid; }
		
		uint32 attributes() const { return KeyAttr; }
		bool attribute(uint32 attr) const { return KeyAttr & attr; }
		void setAttribute(uint32 attr) { KeyAttr |= attr; }
		void clearAttribute(uint32 attr) { KeyAttr &= ~attr; }
		
		uint32 usage() const { return KeyUsage; }
		bool useFor(uint32 u) const { return KeyUsage & u; }

		void usage(uint32 u) { KeyUsage |= u; }
		void clearUsage(uint32 u) { KeyUsage &= ~u; }

    };

	// access to the key header
	Header &header() { return Header::overlay(KeyHeader); }
	const Header &header() const { return Header::overlay(KeyHeader); }
	
	CSSM_KEYBLOB_TYPE blobType() const	{ return header().blobType(); }
	void blobType(CSSM_KEYBLOB_TYPE blobType) { header().blobType(blobType); }

	CSSM_KEYBLOB_FORMAT blobFormat() const { return header().blobFormat(); }
	void blobFormat(CSSM_KEYBLOB_FORMAT blobFormat) { header().blobFormat(blobFormat); }

	CSSM_KEYCLASS keyClass() const		{ return header().keyClass(); }
	void keyClass(CSSM_KEYCLASS keyClass) { header().keyClass(keyClass); }

	CSSM_KEY_TYPE algorithm() const		{ return header().algorithm(); }
	void algorithm(CSSM_KEY_TYPE algorithm) { header().algorithm(algorithm); }

	CSSM_KEY_TYPE wrapAlgorithm() const	{ return header().wrapAlgorithm(); }
	void wrapAlgorithm(CSSM_KEY_TYPE wrapAlgorithm) { header().wrapAlgorithm(wrapAlgorithm); }
    
    CSSM_ENCRYPT_MODE wrapMode() const	{ return header().wrapMode(); }
    void wrapMode(CSSM_ENCRYPT_MODE mode) { header().wrapMode(mode); }
	
	bool isWrapped() const				{ return header().isWrapped(); }
	const Guid &cspGuid() const			{ return header().cspGuid(); }
	
	uint32 attributes() const			{ return header().attributes(); }
	bool attribute(uint32 a) const		{ return header().attribute(a); }
	void setAttribute(uint32 attr) { header().setAttribute(attr); }
	void clearAttribute(uint32 attr) { header().clearAttribute(attr); }

	uint32 usage() const				{ return header().usage(); }
	bool useFor(uint32 u) const			{ return header().useFor(u); }

	void usage(uint32 u) { header().usage(u); }
	void clearUsage(uint32 u) { header().clearUsage(u); }
		
public:
	// access to the key data
	size_t length() const { return KeyData.Length; }
	void *data() const { return KeyData.Data; }
	operator void * () const { return data(); }
	CssmData &keyData()		{ return CssmData::overlay(KeyData); }
	const CssmData &keyData() const { return CssmData::overlay(KeyData); }
	operator CssmData & () { return keyData(); }
	operator const CssmData & () const { return keyData(); }
	operator bool () const { return KeyData.Data != NULL; }
	void operator = (const CssmData &data) { KeyData = data; }
};


//
// Wrapped keys are currently identically structured to normal keys.
// But perhaps in the future...
//
typedef CssmKey CssmWrappedKey;


} // end namespace Security


#endif //_H_CSSMUTILITIES
