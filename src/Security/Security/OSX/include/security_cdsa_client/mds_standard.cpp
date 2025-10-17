/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
// mds_standard - standard-defined MDS record types
//
#include <security_cdsa_client/mds_standard.h>
#include <security_cdsa_client/dlquery.h>


namespace Security {
namespace MDSClient {


//
// CDSA Common relation (one record per module)
//
static const char * const commonAttributes[] = {
	"ModuleID",
	"ModuleName",
	"Path",
	"Desc",
	"DynamicFlag",
	"MultiThreadFlag",
	"ServiceMask",
	NULL
};
Common::Common() : Record(commonAttributes) { }

string Common::moduleID() const					{ return mAttributes[0]; }
string Common::moduleName() const				{ return mAttributes[1]; }
string Common::path() const						{ return mAttributes[2]; }
string Common::description() const				{ return mAttributes[3]; }
bool Common::dynamic() const					{ return mAttributes[4]; }
bool Common::singleThreaded() const				{ return !mAttributes[5]; }
CSSM_SERVICE_MASK Common::serviceMask() const   { return mAttributes[6]; }


//
// Common::Carrier draws in the Common fields for anything with
// a ModuleID attribute (which must be the first attribute listed)
//
Common::Carrier::~Carrier() { }

Common &Common::Carrier::common() const
{
	if (!mCommon) {
		const CssmDbRecordAttributeData &attrs
			= dynamic_cast<const Record *>(this)->attributes();
		RefPointer<Common> rpc;
		rpc = Table<Common>(mds()).fetch(
			Attribute("ModuleID") == string(attrs[0]),
			CSSMERR_DL_ENDOFDATA);
		mCommon = rpc;
	}
	return *mCommon;
}


//
// Attributes that are common to all primary relations
//
static const char * const primaryAttributes[] = {
    "ModuleID",
	"SSID",
	"ModuleName",
	"ProductVersion",
	"Vendor",
	NULL
};
PrimaryRecord::PrimaryRecord(const char * const * names)
	: Record(primaryAttributes)
{
	addAttributes(names);
}

string PrimaryRecord::moduleID() const			{ return mAttributes[0]; }
uint32 PrimaryRecord::subserviceID() const		{ return mAttributes[1]; }
string PrimaryRecord::moduleName() const		{ return mAttributes[2]; }
string PrimaryRecord::productVersion() const	{ return mAttributes[3]; }
string PrimaryRecord::vendor() const			{ return mAttributes[4]; }


//
// CSP Primary relation (one record per CSP SSID)
//
static const char * const cspAttributes[] = {
	// up to Vendor is handled by PrimaryRecord
	"CspType",
	"CspFlags",
	NULL
};
CSP::CSP() : PrimaryRecord(cspAttributes) { }

uint32 CSP::cspType() const						{ return mAttributes[5]; }
CSSM_CSP_FLAGS CSP::cspFlags() const			{ return mAttributes[6]; }


//
// CSP capabilities relation
//
static const char * const capAttributes[] = {
	"ModuleID",
	"SSID",
	"ContextType",
	"AlgType",
	"GroupId",
	"AttributeType",
	"Description",
	NULL
};
CSPCapabilities::CSPCapabilities() : Record(capAttributes) { }

string CSPCapabilities::moduleID() const		{ return mAttributes[0]; }
uint32 CSPCapabilities::subserviceID() const	{ return mAttributes[1]; }
uint32 CSPCapabilities::contextType() const		{ return mAttributes[2]; }
uint32 CSPCapabilities::algorithm() const		{ return mAttributes[3]; }
uint32 CSPCapabilities::group() const			{ return mAttributes[4]; }
uint32 CSPCapabilities::attribute() const		{ return mAttributes[5]; }
string CSPCapabilities::description() const		{ return mAttributes[6]; }


//
// CSP SmartcardInfo relation (one record per smartcard token present)
//
static const char * const scAttributes[] = {
    "ModuleID",
	"SSID",
	"ScDesc",
	"ScVendor",
	"ScVersion",
	"ScFirmwareVersion",
	"ScFlags",
	"ScCustomFlags",
	"ScSerialNumber",
	NULL
};
SmartcardInfo::SmartcardInfo() : Record(scAttributes) { }

string SmartcardInfo::moduleID() const			{ return mAttributes[0]; }
uint32 SmartcardInfo::subserviceID() const		{ return mAttributes[1]; }
string SmartcardInfo::description() const		{ return mAttributes[2]; }
string SmartcardInfo::vendor() const			{ return mAttributes[3]; }
string SmartcardInfo::version() const			{ return mAttributes[4]; }
string SmartcardInfo::firmware() const			{ return mAttributes[5]; }
CSSM_SC_FLAGS SmartcardInfo::flags() const		{ return mAttributes[6]; }
CSSM_SC_FLAGS SmartcardInfo::customFlags() const { return mAttributes[7]; }
string SmartcardInfo::serial() const			{ return mAttributes[8]; }


//
// DL Primary relation (one record per DL SSID)
//
static const char * const dlAttributes[] = {
	// up to Vendor is handled by PrimaryRecord
	"DLType",
	"QueryLimitsFlag",
	NULL
};
DL::DL() : PrimaryRecord(dlAttributes) { }

uint32 DL::dlType() const						{ return mAttributes[5]; }
uint32 DL::queryLimits() const					{ return mAttributes[6]; }


//
// CL Primary relation (one record per CL SSID)
//
static const char * const clAttributes[] = {
	// up to Vendor is handled by PrimaryRecord
	"CertTypeFormat",
	"CrlTypeFormat",
	NULL
};
CL::CL() : PrimaryRecord(clAttributes) { }

uint32 CL::certTypeFormat() const				{ return mAttributes[5]; }
uint32 CL::crlTypeFormat() const				{ return mAttributes[6]; }


//
// TP Primary relation (one record per TP SSID)
//
static const char * const tpAttributes[] = {
	// up to Vendor is handled by PrimaryRecord
	"CertTypeFormat",
	NULL
};
TP::TP() : PrimaryRecord(tpAttributes) { }

uint32 TP::certTypeFormat() const				{ return mAttributes[5]; }


//
// TP Policy-OIDS relation (one record per supported policy and TP)
//
static const char * const policyAttributes[] = {
    "ModuleID",
	"SSID",
	"OID",
	"Value",
	NULL
};
PolicyOids::PolicyOids() : Record(policyAttributes) { }

string PolicyOids::moduleID() const			{ return mAttributes[0]; }
uint32 PolicyOids::subserviceID() const		{ return mAttributes[1]; }
CssmData PolicyOids::oid() const			{ return mAttributes[2]; }
CssmData PolicyOids::value() const			{ return mAttributes[3]; }


} // end namespace MDSClient
} // end namespace Security
