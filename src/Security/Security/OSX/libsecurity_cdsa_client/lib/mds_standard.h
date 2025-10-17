/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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
// mds_standard - standard-defined MDS record types.
//
// These are the C++ record types corresponding to standard and Apple-defined
// MDS relations. Note that not all standard fields are included; only those
// of particular interest to the implementation. Feel free to add field functions
// as needed.
//

#ifndef _H_CDSA_CLIENT_MDS_STANDARD
#define _H_CDSA_CLIENT_MDS_STANDARD

#include <security_cdsa_client/mdsclient.h>


namespace Security {
namespace MDSClient {


//
// The CDSA Common table (one record per module)
//
class Common : public Record {
public:
	Common();
	static const CSSM_DB_RECORDTYPE recordType = MDS_CDSADIR_COMMON_RECORDTYPE;
	
	string moduleID() const;
	string moduleName() const;
	string path() const;
	string description() const;
	bool dynamic() const;
	bool singleThreaded() const;
	CSSM_SERVICE_MASK serviceMask() const;
	
public:
	//
	// "Link in" a Common into another record, whose attributes()[0] is the ModuleID
	//
	class Carrier {
	public:
		virtual ~Carrier();
		
		string moduleName() const			{ return common().moduleName(); }
		string path() const					{ return common().path(); }
		string description() const			{ return common().description(); }
		bool dynamic() const				{ return common().dynamic(); }
		bool singleThreaded() const			{ return common().singleThreaded(); }
		CSSM_SERVICE_MASK serviceMask() const { return common().serviceMask(); }
	
	private:
		mutable RefPointer<Common> mCommon;
		
		Common &common() const;
	};
};


//
// PrimaryRecord shapes the "common head" of all MDS primary relations
//
class PrimaryRecord : public Record, public Common::Carrier {
public:
	PrimaryRecord(const char * const * names);

	string moduleID() const;
	uint32 subserviceID() const;
	string moduleName() const;
	string productVersion() const;
	string vendor() const;
};


//
// The CSP Primary relation
//
class CSP : public PrimaryRecord {
public:
	CSP();
	static const CSSM_DB_RECORDTYPE recordType = MDS_CDSADIR_CSP_PRIMARY_RECORDTYPE;

	uint32 cspType() const;
	CSSM_CSP_FLAGS cspFlags() const;
};


//
// The CSP Capabilities relation
//
class CSPCapabilities : public Record, public Common::Carrier {
public:
	CSPCapabilities();
	static const CSSM_DB_RECORDTYPE recordType = MDS_CDSADIR_CSP_CAPABILITY_RECORDTYPE;

	string moduleID() const;
	uint32 subserviceID() const;
	uint32 contextType() const;
	uint32 algorithm() const;
	uint32 group() const;
	uint32 attribute() const;
	string description() const;
};


//
// The CSP "smartcard token" relation
//
class SmartcardInfo : public Record, public Common::Carrier {
public:
	SmartcardInfo();
	static const CSSM_DB_RECORDTYPE recordType = MDS_CDSADIR_CSP_SC_INFO_RECORDTYPE;
	
	string moduleID() const;
	uint32 subserviceID() const;
	string description() const;
	string vendor() const;
	string version() const;
	string firmware() const;
	CSSM_SC_FLAGS flags() const;
	CSSM_SC_FLAGS customFlags() const;
	string serial() const;
};


//
// The DL Primary relation
//
class DL : public PrimaryRecord {
public:
	DL();
	static const CSSM_DB_RECORDTYPE recordType = MDS_CDSADIR_DL_PRIMARY_RECORDTYPE;

	uint32 dlType() const;
	uint32 queryLimits() const;
};


//
// The CL Primary relation
//
class CL : public PrimaryRecord {
public:
	CL();
	static const CSSM_DB_RECORDTYPE recordType = MDS_CDSADIR_CL_PRIMARY_RECORDTYPE;

	uint32 certTypeFormat() const;
	 uint32 certType() const { return certTypeFormat() >> 16; }
	 uint32 certEncoding() const { return certTypeFormat() & 0xFFFF; }
	uint32 crlTypeFormat() const;
	 uint32 crlType() const { return crlTypeFormat() >> 16; }
	 uint32 crlEncoding() const { return crlTypeFormat() & 0xFFFF; }
};


//
// The TP Primary relation
//
class TP : public PrimaryRecord {
public:
	TP();
	static const CSSM_DB_RECORDTYPE recordType = MDS_CDSADIR_TP_PRIMARY_RECORDTYPE;

	uint32 certTypeFormat() const;
	 uint32 certType() const { return certTypeFormat() >> 16; }
	 uint32 certEncoding() const { return certTypeFormat() & 0xFFFF; }
};


//
// The TP Policy-OIDS relation
//
class PolicyOids : public Record {
public:
	PolicyOids();
	static const CSSM_DB_RECORDTYPE recordType = MDS_CDSADIR_TP_OIDS_RECORDTYPE;
	
	string moduleID() const;
	uint32 subserviceID() const;
	CssmData oid() const;
	CssmData value() const;
};


} // end namespace MDSClient
} // end namespace Security

#endif // _H_CDSA_CLIENT_MDS_STANDARD
