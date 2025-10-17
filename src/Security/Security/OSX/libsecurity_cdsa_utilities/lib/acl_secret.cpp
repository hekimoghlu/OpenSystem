/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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
// acl_secret - secret-validation password ACLs framework.
//
#include <security_cdsa_utilities/acl_secret.h>
#include <security_utilities/trackingallocator.h>
#include <security_utilities/debugging.h>
#include <security_utilities/endian.h>
#include <algorithm>


//
// Construct a secret-bearing ACL subject
//
SecretAclSubject::SecretAclSubject(Allocator &alloc,
		CSSM_ACL_SUBJECT_TYPE type, const CssmData &password)
	: SimpleAclSubject(type), allocator(alloc),
	  mSecret(alloc, password), mSecretValid(true), mCacheSecret(false)
{ }

SecretAclSubject::SecretAclSubject(Allocator &alloc,
		CSSM_ACL_SUBJECT_TYPE type, CssmManagedData &password)
    : SimpleAclSubject(type), allocator(alloc),
	  mSecret(alloc, password), mSecretValid(true), mCacheSecret(false)
{ }

SecretAclSubject::SecretAclSubject(Allocator &alloc,
		CSSM_ACL_SUBJECT_TYPE type, bool doCache)
	: SimpleAclSubject(type), allocator(alloc),
	  mSecret(alloc), mSecretValid(false), mCacheSecret(doCache)
{ }


//
// Set the secret after creation.
//
// These are const methods by design, even though they obvious (may) set
// a field in the SecretAclSubject. The fields are mutable, following the
// general convention that transient state in AclSubjects is mutable.
//
void SecretAclSubject::secret(const CssmData &s) const
{
	assert(!mSecretValid);	// can't re-set it
	if (mCacheSecret) {
		mSecret = s;
		mSecretValid = true;
		secinfo("aclsecret", "%p secret stored", this);
	} else
		secinfo("aclsecret", "%p refused to store secret", this);
}

void SecretAclSubject::secret(CssmManagedData &s) const
{
	assert(!mSecretValid);	// can't re-set it
	if (mCacheSecret) {
		mSecret = s;
		mSecretValid = true;
		secinfo("aclsecret", "%p secret stored", this);
	} else
		secinfo("aclsecret", "%p refused to store secret", this);
}


//
// Validate a secret.
// The subclass has to come up with the secret somehow. We just validate it.
//
bool SecretAclSubject::validates(const AclValidationContext &context,
    const TypedList &sample) const
{
	CssmAutoData secret(allocator);
	
	// try to get the secret; fail if we can't
	if (!getSecret(context, sample, secret))
		return false;
	
	// now validate the secret
	if (mSecretValid) {
		return mSecret == secret;
	} else if (Environment *env = context.environment<Environment>()) {
		TrackingAllocator alloc(Allocator::standard());
		TypedList data(alloc, type(), new(alloc) ListElement(secret.get()));
		CssmSample sample(data);
		AccessCredentials cred((SampleGroup(sample)), context.credTag());
		return env->validateSecret(this, &cred);
	} else {
		return false;
	}
}


#ifdef DEBUGDUMP

void SecretAclSubject::debugDump() const
{
	if (mSecretValid) {
		Debug::dump(" ");
		Debug::dumpData(mSecret.data(), mSecret.length());
	}
	if (mCacheSecret)
		Debug::dump("; CACHING");
}

#endif //DEBUGDUMP
