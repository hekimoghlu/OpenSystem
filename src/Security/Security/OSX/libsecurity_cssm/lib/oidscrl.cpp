/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

 File:      oidscrl.cpp

 Contains:  Object Identifiers for X509 CRLs and OCSP

 Copyright (c) 1999,2001-2002,2004,2011,2014 Apple Inc. All Rights Reserved.

 */

#include <Security/oidscrl.h>
 
static const uint8

	/* CRL OIDs */
	X509V2CRLSignedCrlStruct[]					= {INTEL_X509V2_CRL_R08, 0},
	X509V2CRLSignedCrlCStruct	[]				= {INTEL_X509V2_CRL_R08, 0, INTEL_X509_C_DATATYPE},
	X509V2CRLTbsCertListStruct	[]				= {INTEL_X509V2_CRL_R08, 1},
	X509V2CRLTbsCertListCStruct[]				= {INTEL_X509V2_CRL_R08, 1, INTEL_X509_C_DATATYPE},
	X509V2CRLVersion	[]						= {INTEL_X509V2_CRL_R08, 2},
	X509V1CRLIssuerStruct[]						= {INTEL_X509V2_CRL_R08, 3},
	X509V1CRLIssuerNameCStruct[]				= {INTEL_X509V2_CRL_R08, 3, INTEL_X509_C_DATATYPE},
	X509V1CRLIssuerNameLDAP[]					= {INTEL_X509V2_CRL_R08, 3, 
													INTEL_X509_LDAPSTRING_DATATYPE},
	X509V1CRLThisUpdate[]						= {INTEL_X509V2_CRL_R08, 4},
	X509V1CRLNextUpdate[]						= {INTEL_X509V2_CRL_R08, 5},
	
	/* CRL Entry (CRL CertList) OIDS */
	X509V1CRLRevokedCertificatesStruct[]		= {INTEL_X509V2_CRL_R08, 7},
	X509V1CRLRevokedCertificatesCStruct[]		= {INTEL_X509V2_CRL_R08, 7, INTEL_X509_C_DATATYPE},
	X509V1CRLNumberOfRevokedCertEntries[]		= {INTEL_X509V2_CRL_R08, 6},
	X509V1CRLRevokedEntryStruct[]				= {INTEL_X509V2_CRL_R08, 15},
	X509V1CRLRevokedEntryCStruct[]				= {INTEL_X509V2_CRL_R08, 15, INTEL_X509_C_DATATYPE},
	X509V1CRLRevokedEntrySerialNumber[]			= {INTEL_X509V2_CRL_R08, 16},
	X509V1CRLRevokedEntryRevocationDate[]		= {INTEL_X509V2_CRL_R08, 17},
	
	/* CRL Entry (CRL CertList) Extension OIDs */
	X509V2CRLRevokedEntryAllExtensionsStruct[]	= {INTEL_X509V2_CRL_R08, 18},
	X509V2CRLRevokedEntryAllExtensionsCStruct[]	= {INTEL_X509V2_CRL_R08, 18, INTEL_X509_C_DATATYPE},
	X509V2CRLRevokedEntryNumberOfExtensions[]	= {INTEL_X509V2_CRL_R08, 20},
	X509V2CRLRevokedEntrySingleExtensionStruct[]= {INTEL_X509V2_CRL_R08, 19},
	X509V2CRLRevokedEntrySingleExtensionCStruct[]= {INTEL_X509V2_CRL_R08, 19, INTEL_X509_C_DATATYPE},
	X509V2CRLRevokedEntryExtensionId[]			= {INTEL_X509V2_CRL_R08, 21},
	X509V2CRLRevokedEntryExtensionCritical[]	= {INTEL_X509V2_CRL_R08, 22},
	X509V2CRLRevokedEntryExtensionType[]		= {INTEL_X509V2_CRL_R08, 23},
	X509V2CRLRevokedEntryExtensionValue[]		= {INTEL_X509V2_CRL_R08, 24},
	
	/* CRL Extension OIDs */
	X509V2CRLAllExtensionsStruct[]				= {INTEL_X509V2_CRL_R08, 8},
	X509V2CRLAllExtensionsCStruct[]				= {INTEL_X509V2_CRL_R08, 8, INTEL_X509_C_DATATYPE},
	X509V2CRLNumberOfExtensions[]				= {INTEL_X509V2_CRL_R08, 10},
	X509V2CRLSingleExtensionStruct[]			= {INTEL_X509V2_CRL_R08, 9},
	X509V2CRLSingleExtensionCStruct[]			= {INTEL_X509V2_CRL_R08, 9, INTEL_X509_C_DATATYPE},
	X509V2CRLExtensionId[]						= {INTEL_X509V2_CRL_R08, 11},
	X509V2CRLExtensionCritical[]				= {INTEL_X509V2_CRL_R08, 12},
	X509V2CRLExtensionType[]					= {INTEL_X509V2_CRL_R08, 13},

	/* OCSP */
	OID_PKIX_OCSP[]							= { OID_AD_OCSP },
	OID_PKIX_OCSP_BASIC[]					= { OID_AD_OCSP, 1 },
	OID_PKIX_OCSP_NONCE[]					= { OID_AD_OCSP, 2 },
	OID_PKIX_OCSP_CRL[]						= { OID_AD_OCSP, 3 },
	OID_PKIX_OCSP_RESPONSE[]				= { OID_AD_OCSP, 4 },
	OID_PKIX_OCSP_NOCHECK[]					= { OID_AD_OCSP, 5 },
	OID_PKIX_OCSP_ARCHIVE_CUTOFF[]			= { OID_AD_OCSP, 6 },
	OID_PKIX_OCSP_SERVICE_LOCATOR[]			= { OID_AD_OCSP, 7 };
	
const CSSM_OID

	/* CRL OIDs */
	CSSMOID_X509V2CRLSignedCrlStruct 			= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLSignedCrlStruct},
	CSSMOID_X509V2CRLSignedCrlCStruct 			= {INTEL_X509V2_CRL_R08_LENGTH+2, 
													(uint8 *)X509V2CRLSignedCrlCStruct},
	CSSMOID_X509V2CRLTbsCertListStruct 			= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLTbsCertListStruct},
	CSSMOID_X509V2CRLTbsCertListCStruct 		= {INTEL_X509V2_CRL_R08_LENGTH+2, 
													(uint8 *)X509V2CRLTbsCertListCStruct},
	CSSMOID_X509V2CRLVersion 					= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLVersion},
	CSSMOID_X509V1CRLIssuerStruct 				= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V1CRLIssuerStruct},
	CSSMOID_X509V1CRLIssuerNameCStruct 			= {INTEL_X509V2_CRL_R08_LENGTH+2, 
													(uint8 *)X509V1CRLIssuerNameCStruct},
	CSSMOID_X509V1CRLIssuerNameLDAP 			= {INTEL_X509V2_CRL_R08_LENGTH+2, 
													(uint8 *)X509V1CRLIssuerNameLDAP},
	CSSMOID_X509V1CRLThisUpdate 				= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V1CRLThisUpdate},
	CSSMOID_X509V1CRLNextUpdate 				= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V1CRLNextUpdate},

	/* CRL Entry (CRL CertList) OIDS */
	CSSMOID_X509V1CRLRevokedCertificatesStruct 	= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V1CRLRevokedCertificatesStruct},
	CSSMOID_X509V1CRLRevokedCertificatesCStruct	= {INTEL_X509V2_CRL_R08_LENGTH+2, 
													(uint8 *)X509V1CRLRevokedCertificatesCStruct},
	CSSMOID_X509V1CRLNumberOfRevokedCertEntries	= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V1CRLNumberOfRevokedCertEntries},
	CSSMOID_X509V1CRLRevokedEntryStruct 		= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V1CRLRevokedEntryStruct},
	CSSMOID_X509V1CRLRevokedEntryCStruct 		= {INTEL_X509V2_CRL_R08_LENGTH+2, 
													(uint8 *)X509V1CRLRevokedEntryCStruct},
	CSSMOID_X509V1CRLRevokedEntrySerialNumber 	= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V1CRLRevokedEntrySerialNumber},
	CSSMOID_X509V1CRLRevokedEntryRevocationDate	= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V1CRLRevokedEntryRevocationDate},

	/* CRL Entry (CRL CertList) Extension OIDs */
	CSSMOID_X509V2CRLRevokedEntryAllExtensionsStruct 	= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLRevokedEntryAllExtensionsStruct},
	CSSMOID_X509V2CRLRevokedEntryAllExtensionsCStruct 	= {INTEL_X509V2_CRL_R08_LENGTH+2, 
													(uint8 *)X509V2CRLRevokedEntryAllExtensionsCStruct},
	CSSMOID_X509V2CRLRevokedEntryNumberOfExtensions 	= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLRevokedEntryNumberOfExtensions},
	CSSMOID_X509V2CRLRevokedEntrySingleExtensionStruct 	= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLRevokedEntrySingleExtensionStruct},
	CSSMOID_X509V2CRLRevokedEntrySingleExtensionCStruct = {INTEL_X509V2_CRL_R08_LENGTH+2, 
													(uint8 *)X509V2CRLRevokedEntrySingleExtensionCStruct},
	CSSMOID_X509V2CRLRevokedEntryExtensionId 			= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLRevokedEntryExtensionId},
	CSSMOID_X509V2CRLRevokedEntryExtensionCritical 		= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLRevokedEntryExtensionCritical},
	CSSMOID_X509V2CRLRevokedEntryExtensionType 			= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLRevokedEntryExtensionType},
	CSSMOID_X509V2CRLRevokedEntryExtensionValue 		= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													 (uint8 *)X509V2CRLRevokedEntryExtensionValue},

	/* CRL Extension OIDs */
	CSSMOID_X509V2CRLAllExtensionsStruct 		= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLAllExtensionsStruct},
	CSSMOID_X509V2CRLAllExtensionsCStruct 		= {INTEL_X509V2_CRL_R08_LENGTH+2, 
													(uint8 *)X509V2CRLAllExtensionsCStruct},
	CSSMOID_X509V2CRLNumberOfExtensions 		= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLNumberOfExtensions},
	CSSMOID_X509V2CRLSingleExtensionStruct 		= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLSingleExtensionStruct},
	CSSMOID_X509V2CRLSingleExtensionCStruct 	= {INTEL_X509V2_CRL_R08_LENGTH+2, 
													(uint8 *)X509V2CRLSingleExtensionCStruct},
	CSSMOID_X509V2CRLExtensionId 				= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLExtensionId},
	CSSMOID_X509V2CRLExtensionCritical 			= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLExtensionCritical},
	CSSMOID_X509V2CRLExtensionType 				= {INTEL_X509V2_CRL_R08_LENGTH+1, 
													(uint8 *)X509V2CRLExtensionType};

const CSSM_OID
	/* OCSP OIDs */
	CSSMOID_PKIX_OCSP						= { OID_AD_OCSP_LENGTH, (uint8 *)OID_PKIX_OCSP },
	CSSMOID_PKIX_OCSP_BASIC					= { OID_AD_OCSP_LENGTH+1, (uint8 *)OID_PKIX_OCSP_BASIC},
	CSSMOID_PKIX_OCSP_NONCE					= { OID_AD_OCSP_LENGTH+1, (uint8 *)OID_PKIX_OCSP_NONCE},
	CSSMOID_PKIX_OCSP_CRL					= { OID_AD_OCSP_LENGTH+1, (uint8 *)OID_PKIX_OCSP_CRL},
	CSSMOID_PKIX_OCSP_RESPONSE				= { OID_AD_OCSP_LENGTH+1, (uint8 *)OID_PKIX_OCSP_RESPONSE},
	CSSMOID_PKIX_OCSP_NOCHECK				= { OID_AD_OCSP_LENGTH+1, (uint8 *)OID_PKIX_OCSP_NOCHECK},
	CSSMOID_PKIX_OCSP_ARCHIVE_CUTOFF		= { OID_AD_OCSP_LENGTH+1, (uint8 *)OID_PKIX_OCSP_ARCHIVE_CUTOFF},
	CSSMOID_PKIX_OCSP_SERVICE_LOCATOR		= { OID_AD_OCSP_LENGTH+1, (uint8 *)OID_PKIX_OCSP_SERVICE_LOCATOR};
