/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
// Schema.h
//
#ifndef _SECURITY_SCHEMA_H_
#define _SECURITY_SCHEMA_H_

#include <Security/SecKeychainItem.h>

namespace Security {

namespace KeychainCore {

namespace Schema {

CSSM_DB_RECORDTYPE recordTypeFor(SecItemClass itemClass);
SecItemClass itemClassFor(CSSM_DB_RECORDTYPE recordType);
bool haveAttributeInfo(SecKeychainAttrType attrType);
const CSSM_DB_ATTRIBUTE_INFO &attributeInfo(SecKeychainAttrType attrType);

extern const CSSM_DB_ATTRIBUTE_INFO RelationID;
extern const CSSM_DB_ATTRIBUTE_INFO RelationName;
extern const CSSM_DB_ATTRIBUTE_INFO AttributeID;
extern const CSSM_DB_ATTRIBUTE_INFO AttributeNameFormat;
extern const CSSM_DB_ATTRIBUTE_INFO AttributeName;
extern const CSSM_DB_ATTRIBUTE_INFO AttributeNameID;
extern const CSSM_DB_ATTRIBUTE_INFO AttributeFormat;
extern const CSSM_DB_ATTRIBUTE_INFO IndexType;

extern const CSSM_DBINFO DBInfo;

// Certificate attributes and schema
extern const CSSM_DB_ATTRIBUTE_INFO kX509CertificateCertType;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CertificateCertEncoding;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CertificatePrintName;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CertificateAlias;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CertificateSubject;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CertificateIssuer;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CertificateSerialNumber;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CertificateSubjectKeyIdentifier;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CertificatePublicKeyHash;

extern const CSSM_DB_SCHEMA_ATTRIBUTE_INFO X509CertificateSchemaAttributeList[];
extern const CSSM_DB_SCHEMA_INDEX_INFO X509CertificateSchemaIndexList[];
extern const uint32 X509CertificateSchemaAttributeCount;
extern const uint32 X509CertificateSchemaIndexCount;

// CRL attributes and schema
extern const CSSM_DB_ATTRIBUTE_INFO kX509CrlCrlType;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CrlCrlEncoding;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CrlPrintName;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CrlAlias;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CrlIssuer;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CrlSerialNumber;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CrlThisUpdate;
extern const CSSM_DB_ATTRIBUTE_INFO kX509CrlNextUpdate;

extern const CSSM_DB_SCHEMA_ATTRIBUTE_INFO X509CrlSchemaAttributeList[];
extern const CSSM_DB_SCHEMA_INDEX_INFO X509CrlSchemaIndexList[];
extern const uint32 X509CrlSchemaAttributeCount;
extern const uint32 X509CrlSchemaIndexCount;

// UserTrust records attributes and schema
extern const CSSM_DB_ATTRIBUTE_INFO kUserTrustTrustedCertificate;
extern const CSSM_DB_ATTRIBUTE_INFO kUserTrustTrustedPolicy;

extern const CSSM_DB_SCHEMA_ATTRIBUTE_INFO UserTrustSchemaAttributeList[];
extern const CSSM_DB_SCHEMA_INDEX_INFO UserTrustSchemaIndexList[];
extern const uint32 UserTrustSchemaAttributeCount;
extern const uint32 UserTrustSchemaIndexCount;

// UnlockReferral records attributes and schema
extern const CSSM_DB_ATTRIBUTE_INFO kUnlockReferralType;
extern const CSSM_DB_ATTRIBUTE_INFO kUnlockReferralDbName;
extern const CSSM_DB_ATTRIBUTE_INFO kUnlockReferralDbGuid;
extern const CSSM_DB_ATTRIBUTE_INFO kUnlockReferralDbSSID;
extern const CSSM_DB_ATTRIBUTE_INFO kUnlockReferralDbSSType;
extern const CSSM_DB_ATTRIBUTE_INFO kUnlockReferralDbNetname;
extern const CSSM_DB_ATTRIBUTE_INFO kUnlockReferralKeyLabel;
extern const CSSM_DB_ATTRIBUTE_INFO kUnlockReferralKeyAppTag;
extern const CSSM_DB_ATTRIBUTE_INFO kUnlockReferralPrintName;
extern const CSSM_DB_ATTRIBUTE_INFO kUnlockReferralAlias;

extern const CSSM_DB_SCHEMA_ATTRIBUTE_INFO UnlockReferralSchemaAttributeList[];
extern const CSSM_DB_SCHEMA_INDEX_INFO UnlockReferralSchemaIndexList[];
extern const uint32 UnlockReferralSchemaAttributeCount;
extern const uint32 UnlockReferralSchemaIndexCount;

// Extended Attribute record attributes and schema
extern const CSSM_DB_ATTRIBUTE_INFO kExtendedAttributeRecordType;
extern const CSSM_DB_ATTRIBUTE_INFO kExtendedAttributeItemID;
extern const CSSM_DB_ATTRIBUTE_INFO kExtendedAttributeAttributeName;
extern const CSSM_DB_ATTRIBUTE_INFO kExtendedAttributeModDate;
extern const CSSM_DB_ATTRIBUTE_INFO kExtendedAttributeAttributeValue;

extern const CSSM_DB_SCHEMA_ATTRIBUTE_INFO ExtendedAttributeSchemaAttributeList[];
extern const CSSM_DB_SCHEMA_INDEX_INFO ExtendedAttributeSchemaIndexList[];
extern const uint32 ExtendedAttributeSchemaAttributeCount;
extern const uint32 ExtendedAttributeSchemaIndexCount;

} // end namespace Schema

} // end namespace KeychainCore

} // end namespace Security

#endif // !_SECURITY_SCHEMA_H_
