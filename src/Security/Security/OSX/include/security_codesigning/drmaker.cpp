/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
// drmaker - create automatic Designated Requirements
//
#include "drmaker.h"
#include "csutilities.h"
#include <Security/oidsbase.h>
#include <Security/SecCertificatePriv.h>
#include <libDER/oids.h>
//#include <Security/cssmapplePriv.h>

namespace Security {
namespace CodeSigning {


static const uint8_t adcSdkMarker[] = { APPLE_EXTENSION_OID, 2, 1 };		// iOS intermediate marker
const CSSM_DATA adcSdkMarkerOID = { sizeof(adcSdkMarker), (uint8_t *)adcSdkMarker };

static const uint8_t caspianSdkMarker[] = { APPLE_EXTENSION_OID, 2, 6 }; // Caspian intermediate marker
const CSSM_DATA devIdSdkMarkerOID = { sizeof(caspianSdkMarker), (uint8_t *)caspianSdkMarker };
static const uint8_t caspianLeafMarker[] = { APPLE_EXTENSION_OID, 1, 13 }; // Caspian leaf certificate marker
const CSSM_DATA devIdLeafMarkerOID = { sizeof(caspianLeafMarker), (uint8_t *)caspianLeafMarker };



DRMaker::DRMaker(const Requirement::Context &context)
	: ctx(context)
{
}

DRMaker::~DRMaker()
{
}


//
// Generate the default (implicit) Designated Requirement for this StaticCode.
// This is a heuristic of sorts, and may change over time (for the better, we hope).
//
Requirement *DRMaker::make()
{
	// we can't make an explicit DR for a (proposed) ad-hoc signing because that requires the CodeDirectory (which we ain't got yet)
	if (ctx.certCount() == 0)
		return NULL;

	// always require the identifier
	this->put(opAnd);
	this->ident(ctx.identifier);
	
	if (isAppleCA(ctx.cert(Requirement::anchorCert))
#if	defined(TEST_APPLE_ANCHOR)
		|| !memcmp(anchorHash, Requirement::testAppleAnchorHash(), SHA1::digestLength)
#endif
		)
		appleAnchor();
	else
		nonAppleAnchor();
	
	return Maker::make();
}


void DRMaker::nonAppleAnchor()
{
	// get the Organization DN element for the leaf
	CFRef<CFStringRef> leafOrganization = NULL;

	leafOrganization.take(SecCertificateCopySubjectAttributeValue(ctx.cert(Requirement::leafCert), (DERItem *)&oidOrganizationName));
	if (!leafOrganization) {
		secinfo("drmaker", "Unable to get OrganizationName from leaf certificate");
	}

	// now step up the cert chain looking for the first cert with a different one
	int slot = Requirement::leafCert;						// start at leaf
	if (leafOrganization) {
		while (SecCertificateRef ca = ctx.cert(slot+1)) {		// NULL if you over-run the anchor slot
			CFRef<CFStringRef> caOrganization = NULL;

			caOrganization.take(SecCertificateCopySubjectAttributeValue(ca, (DERItem *)&oidOrganizationName));
			if (!caOrganization) {
				secinfo("drmaker", "Unable to get OrganizationName from certificate");
			}

			if (!caOrganization || CFStringCompare(leafOrganization, caOrganization, 0) != kCFCompareEqualTo)
				break;
			slot++;
		}
		if (slot == (int)ctx.certCount() - 1)		// went all the way to the anchor...
			slot = Requirement::anchorCert;					// ... so say that
	}
	
	// nail the last cert with the leaf's Organization value
	SHA1::Digest authorityHash;
	hashOfCertificate(ctx.cert(slot), authorityHash);
	this->anchor(slot, authorityHash);
}


void DRMaker::appleAnchor()
{
	if (isIOSSignature()) {
		// get the Common Name DN element for the leaf
		CFRef<CFStringRef> leafCN = NULL;

		leafCN.take(SecCertificateCopySubjectAttributeValue(ctx.cert(Requirement::leafCert), (DERItem *)&oidCommonName));
		if (!leafCN) {
			secinfo("drmaker", "Unable to get CommonName from leaf certificate");
		}
		
		// apple anchor generic and ...
		this->put(opAnd);
		this->anchorGeneric();			// apple generic anchor and...
		// ... leaf[subject.CN] = <leaf's subject> and ...
		this->put(opAnd);
		this->put(opCertField);			// certificate
		this->put(0);					// leaf
		this->put("subject.CN");		// [subject.CN]
		this->put(matchEqual);			// =
		this->putData(leafCN);			// <leaf CN>
		// ... cert 1[field.<marker>] exists
		this->put(opCertGeneric);		// certificate
		this->put(1);					// 1
		this->putData(adcSdkMarkerOID.Data, adcSdkMarkerOID.Length); // [field.<marker>]
		this->put(matchExists);			// exists
		return;
	}
	
	if (isDeveloperIDSignature()) {
		// get the Organizational Unit DN element for the leaf (it contains the TEAMID)
		CFRef<CFStringRef> teamID = NULL;

		teamID.take(SecCertificateCopySubjectAttributeValue(ctx.cert(Requirement::leafCert), (DERItem *)&oidOrganizationalUnitName));
		if (!teamID) {
			secinfo("drmaker", "Unable to get teamID from leaf certificate");
		}

		// apple anchor generic and ...
		this->put(opAnd);
		this->anchorGeneric();			// apple generic anchor and...
		
		// ... certificate 1[intermediate marker oid] exists and ...
		this->put(opAnd);
		this->put(opCertGeneric);		// certificate
		this->put(1);					// 1
		this->putData(caspianSdkMarker, sizeof(caspianSdkMarker));
		this->put(matchExists);			// exists
		
		// ... certificate leaf[Caspian cert oid] exists and ...
		this->put(opAnd);
		this->put(opCertGeneric);		// certificate
		this->put(0);					// leaf
		this->putData(caspianLeafMarker, sizeof(caspianLeafMarker));
		this->put(matchExists);			// exists

		// ... leaf[subject.OU] = <leaf's subject>
		this->put(opCertField);			// certificate
		this->put(0);					// leaf
		this->put("subject.OU");		// [subject.OU]
		this->put(matchEqual);			// =
		this->putData(teamID);			// TEAMID
		return;
	}

	// otherwise, claim this program for Apple Proper
	this->anchor();
}

bool DRMaker::isIOSSignature()
{
	if (ctx.certCount() == 3)		// leaf, one intermediate, anchor
		if (SecCertificateRef intermediate = ctx.cert(1)) // get intermediate
			if (certificateHasField(intermediate, adcSdkMarkerOID))
				return true;
	return false;
}

bool DRMaker::isDeveloperIDSignature()
{
	if (ctx.certCount() == 3)		// leaf, one intermediate, anchor
		if (SecCertificateRef intermediate = ctx.cert(1)) // get intermediate
			if (certificateHasField(intermediate, devIdSdkMarkerOID))
				return true;
	return false;
}


} // end namespace CodeSigning
} // end namespace Security
