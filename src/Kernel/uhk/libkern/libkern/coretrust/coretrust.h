/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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
#ifndef __CORETRUST_H
#define __CORETRUST_H

#include <os/base.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#if XNU_KERNEL_PRIVATE
/*
 * Only include this when building for XNU. CoreTrust will include its local copy
 * of the header.
 */
#include <CoreTrust/CTEvaluate.h>
#endif

#define XNU_SUPPORTS_CORETRUST_AMFI 1
typedef int (*coretrust_CTEvaluateAMFICodeSignatureCMS_t)(
	const uint8_t *cms_data,
	size_t cms_data_length,
	const uint8_t *detached_data,
	size_t detached_data_length,
	bool allow_test_hierarchy,
	const uint8_t **leaf_certificate,
	size_t *leaf_certificate_length,
	CoreTrustPolicyFlags *policy_flags,
	CoreTrustDigestType *cms_digest_type,
	CoreTrustDigestType *hash_agility_digest_type,
	const uint8_t **digest_data,
	size_t *digest_length
	);

#define XNU_SUPPORTS_CORETRUST_LOCAL_SIGNING 1
typedef int (*coretrust_CTEvaluateAMFICodeSignatureCMSPubKey_t)(
	const uint8_t *cms_data,
	size_t cms_data_length,
	const uint8_t *detached_data,
	size_t detached_data_length,
	const uint8_t *anchor_public_key,
	size_t anchor_public_key_length,
	CoreTrustDigestType *cms_digest_type,
	CoreTrustDigestType *hash_agility_digest_type,
	const uint8_t **digest_data,
	size_t *digest_length
	);

#define XNU_SUPPORTS_CORETRUST_PROVISIONING_PROFILE 1
typedef int (*coretrust_CTEvaluateProvisioningProfile_t)(
	const uint8_t *provisioning_profile_data,
	size_t provisioning_profile_length,
	bool allow_test_roots,
	const uint8_t **profile_content,
	size_t *profile_content_length
	);

#define XNU_SUPPORTS_CORETRUST_MULTI_STEP_AMFI 1

typedef int (*coretrust_CTParseKey_t)(
	const uint8_t *cert_data,
	size_t cert_length,
	const uint8_t **key_data,
	size_t *key_length
	);

typedef int (*coretrust_CTParseAmfiCMS_t)(
	const uint8_t *cms_data,
	size_t cms_length,
	CoreTrustDigestType max_digest_type,
	const uint8_t **leaf_cert, size_t *leaf_cert_length,
	const uint8_t **content_data, size_t *content_length,
	CoreTrustDigestType *cms_digest_type,
	CoreTrustPolicyFlags *policy_flags
	);

typedef int (*coretrust_CTVerifyAmfiCMS_t)(
	const uint8_t *cms_data,
	size_t cms_length,
	const uint8_t *digest_data,
	size_t digest_length,
	CoreTrustDigestType max_digest_type,
	CoreTrustDigestType *hash_agility_digest_type,
	const uint8_t **agility_digest_data,
	size_t *agility_digest_length
	);

typedef int (*coretrust_CTVerifyAmfiCertificateChain_t)(
	const uint8_t *cms_data,
	size_t cms_length,
	bool allow_test_hierarchy,
	CoreTrustDigestType max_digest_type,
	CoreTrustPolicyFlags *policy_flags
	);

typedef struct _coretrust {
	coretrust_CTEvaluateAMFICodeSignatureCMS_t CTEvaluateAMFICodeSignatureCMS;
	coretrust_CTEvaluateAMFICodeSignatureCMSPubKey_t CTEvaluateAMFICodeSignatureCMSPubKey;
	coretrust_CTEvaluateProvisioningProfile_t CTEvaluateProvisioningProfile;
	coretrust_CTParseKey_t CTParseKey;
	coretrust_CTParseAmfiCMS_t CTParseAmfiCMS;
	coretrust_CTVerifyAmfiCMS_t CTVerifyAmfiCMS;
	coretrust_CTVerifyAmfiCertificateChain_t CTVerifyAmfiCertificateChain;
} coretrust_t;

__BEGIN_DECLS

/*!
 * @const coretrust_appstore_policy
 * The CoreTrust policy flags which collectively map an applications
 * signature to the App Store certificate chain.
 */
static const CoreTrustPolicyFlags coretrust_appstore_policy =
    CORETRUST_POLICY_IPHONE_APP_PROD  | CORETRUST_POLICY_IPHONE_APP_DEV |
    CORETRUST_POLICY_XROS_APP_PROD    | CORETRUST_POLICY_XROS_APP_DEV   |
    CORETRUST_POLICY_TVOS_APP_PROD    | CORETRUST_POLICY_TVOS_APP_DEV   |
    CORETRUST_POLICY_TEST_FLIGHT_PROD | CORETRUST_POLICY_TEST_FLIGHT_DEV;

/*!
 * @const coretrust_profile_validated_policy
 * The CoreTrust policy flags which collectively map an applications
 * signature to the profile validated certificate chain.
 */
static const CoreTrustPolicyFlags coretrust_profile_validated_policy =
    CORETRUST_POLICY_IPHONE_DEVELOPER | CORETRUST_POLICY_IPHONE_DISTRIBUTION;

/*!
 * @const coretrust_local_signing_policy
 * The CoreTrust policy which maps an application's signature to the locally
 * signed key.
 */
static const CoreTrustPolicyFlags coretrust_local_signing_policy =
    CORETRUST_POLICY_BASIC;

/*!
 * @const coretrust_provisioning_profile_policy
 * The CoreTrust policy which maps a profile's signature to the provisioning
 * profile WWDR certificate chain.
 */
static const CoreTrustPolicyFlags coretrust_provisioning_profile_policy =
    CORETRUST_POLICY_PROVISIONING_PROFILE;

/*!
 * @const coretrust
 * The CoreTrust interface that was registered.
 */
extern const coretrust_t *coretrust;

/*!
 * @function coretrust_interface_register
 * Registers the CoreTrust kext interface for use within the kernel proper.
 *
 * @param ct
 * The interface to register.
 *
 * @discussion
 * This routine may only be called once and must be called before late-const has
 * been applied to kernel memory.
 */
OS_EXPORT OS_NONNULL1
void
coretrust_interface_register(const coretrust_t *ct);

__END_DECLS

#endif // __CORETRUST_H
