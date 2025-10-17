/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
#ifndef _SECURITY_H_
#define _SECURITY_H_

#include <Security/SecBase.h>
#include <Security/SecCertificate.h>
#include <Security/SecIdentity.h>
#include <Security/SecAccessControl.h>
#include <Security/SecItem.h>
#include <Security/SecKey.h>
#include <Security/SecPolicy.h>
#include <Security/SecRandom.h>
#include <Security/SecImportExport.h>
#include <Security/SecTrust.h>
#include <Security/SecSharedCredential.h>
#include <Security/SecProtocolOptions.h>
#include <Security/SecProtocolMetadata.h>

#if SEC_OS_OSX_INCLUDES
#include <Security/AuthSession.h>
#endif

#if SEC_OS_OSX_INCLUDES
/* CDSA */
#include <Security/cssmconfig.h>
#include <Security/cssmapple.h>
#include <Security/certextensions.h>
#include <Security/cssm.h>
#include <Security/cssmaci.h>
#include <Security/cssmapi.h>
#include <Security/cssmcli.h>
#include <Security/cssmcspi.h>
#include <Security/cssmdli.h>
#include <Security/cssmerr.h>
#include <Security/cssmkrapi.h>
#include <Security/cssmkrspi.h>
#include <Security/cssmspi.h>
#include <Security/cssmtpi.h>
#include <Security/cssmtype.h>
#include <Security/emmspi.h>
#include <Security/emmtype.h>
#include <Security/mds.h>
#include <Security/mds_schema.h>
#include <Security/oidsalg.h>
#include <Security/oidsattr.h>
#include <Security/oidsbase.h>
#include <Security/oidscert.h>
#include <Security/oidscrl.h>
#include <Security/x509defs.h>

/* Security */
#include <Security/SecAccess.h>
#include <Security/SecACL.h>
#include <Security/SecCertificateOIDs.h>
#include <Security/SecIdentitySearch.h>
#include <Security/SecKeychain.h>
#include <Security/SecKeychainItem.h>
#include <Security/SecKeychainSearch.h>
#include <Security/SecPolicySearch.h>
#include <Security/SecTrustedApplication.h>
#include <Security/SecTrustSettings.h>

/* Code Signing */
#include <Security/SecStaticCode.h>
#include <Security/SecCode.h>
#include <Security/SecCodeHost.h>
#include <Security/SecRequirement.h>
#include <Security/SecTask.h>

/* Authorization */
#include <Security/Authorization.h>
#include <Security/AuthorizationTags.h>
#include <Security/AuthorizationDB.h>

/* CMS */
#include <Security/CMSDecoder.h>
#include <Security/CMSEncoder.h>

/* Secure Transport */
#include <Security/CipherSuite.h>
#include <Security/SecureTransport.h>

#ifdef __BLOCKS__
#include <Security/SecTransform.h>
#include <Security/SecCustomTransform.h>
#include <Security/SecDecodeTransform.h>
#include <Security/SecDigestTransform.h>
#include <Security/SecEncodeTransform.h>
#include <Security/SecEncryptTransform.h>
#include <Security/SecSignVerifyTransform.h>
#include <Security/SecReadTransform.h>
#endif

/* DER */
#include <Security/oids.h>

#endif // SEC_OS_OSX

#endif // _SECURITY_H_
