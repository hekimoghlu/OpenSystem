/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
// localkey - Key objects that store a local CSSM key object
//
#include "localkey.h"
#include "server.h"
#include "database.h"
#include <security_cdsa_utilities/acl_any.h>
#include <security_utilities/cfmunge.h>
#include <security_utilities/logging.h>


//
// Create a Key from an explicit CssmKey.
//
LocalKey::LocalKey(Database &db, const CssmKey &newKey, CSSM_KEYATTR_FLAGS moreAttributes)
	: Key(db), mDigest(Server::csp().allocator())
{
	mValidKey = true;
	setup(newKey, moreAttributes);
    secinfo("SSkey", "%p (handle %#x) created from key alg=%u use=0x%x attr=0x%x db=%p",
        this, handle(), mKey.header().algorithm(), mKey.header().usage(), mAttributes, &db);
}


LocalKey::LocalKey(Database &db, CSSM_KEYATTR_FLAGS attributes)
	: Key(db), mValidKey(false), mAttributes(attributes), mDigest(Server::csp().allocator())
{
}


//
// Set up the CssmKey part of this Key according to instructions.
//
void LocalKey::setup(const CssmKey &newKey, CSSM_KEYATTR_FLAGS moreAttributes)
{
	mKey = CssmClient::Key(Server::csp(), newKey, false);
    CssmKey::Header &header = mKey->header();
    
	// copy key header
	header = newKey.header();
    mAttributes = (header.attributes() & ~forcedAttributes) | moreAttributes;
	
	// apply initial values of derived attributes (these are all in managedAttributes)
    if (!(mAttributes & CSSM_KEYATTR_EXTRACTABLE))
        mAttributes |= CSSM_KEYATTR_NEVER_EXTRACTABLE;
    if (mAttributes & CSSM_KEYATTR_SENSITIVE)
        mAttributes |= CSSM_KEYATTR_ALWAYS_SENSITIVE;

    // verify internal/external attribute separation
    assert((header.attributes() & managedAttributes) == forcedAttributes);
}


LocalKey::~LocalKey()
{
    secinfo("SSkey", "%p destroyed", this);
}


void LocalKey::setOwner(const AclEntryPrototype *owner)
{
	// establish initial ACL; reinterpret empty (null-list) owner as NULL for resilence's sake
	if (owner && !owner->subject().empty())
		acl().cssmSetInitial(*owner);					// specified
	else
		acl().cssmSetInitial(new AnyAclSubject());		// defaulted

    if (this->database().dbVersion() >= CommonBlob::version_partition) {
        // put payload into an AclEntry tagged as CSSM_APPLE_ACL_TAG_PARTITION_ID...
        // ... unless the client has the "converter" entitlement as attested by Apple
        if (!(process().checkAppleSigned() && process().hasEntitlement(migrationEntitlement)))
            this->acl().createClientPartitionID(this->process());
    }
}


LocalDatabase &LocalKey::database() const
{
	return referent<LocalDatabase>();
}


//
// Retrieve the actual CssmKey value for the key object.
// This will decode its blob if needed (and appropriate).
//
CssmClient::Key LocalKey::keyValue()
{
	StLock<Mutex> _(*this);
    if (!mValidKey) {
		getKey();
		mValidKey = true;
	}
    return mKey;
}


//
// Return external key attributees
//
CSSM_KEYATTR_FLAGS LocalKey::attributes()
{
	return mAttributes;
}


//
// Return a key's handle and header in external form
//
void LocalKey::returnKey(U32HandleObject::Handle &h, CssmKey::Header &hdr)
{
	StLock<Mutex> _(*this);

    // return handle
    h = this->handle();
	
	// obtain the key header, from the valid key or the blob if no valid key
	if (mValidKey) {
		hdr = mKey.header();
	} else {
		getHeader(hdr);
	}
    
    // adjust for external attributes
	hdr.clearAttribute(forcedAttributes);
    hdr.setAttribute(mAttributes);
}


//
// Generate the canonical key digest.
// This is defined by a CSP feature that we invoke here.
//
const CssmData &LocalKey::canonicalDigest()
{
	StLock<Mutex> _(*this);
	if (!mDigest) {
		CssmClient::PassThrough ctx(Server::csp());
		ctx.key(keyValue());
		CssmData *digest = NULL;
		ctx(CSSM_APPLECSP_KEYDIGEST, (const void *)NULL, &digest);
		assert(digest);
		mDigest.set(*digest);	// takes ownership of digest data
		Server::csp().allocator().free(digest);	// the CssmData itself
	}
	return mDigest.get();
}

void LocalKey::publicKey(const Context &context, CssmData &pubKeyData)
{
    CssmClient::PassThrough ctx(Server::csp());
    ctx.key(keyValue());
    CSSM_KEYBLOB_FORMAT format = CSSM_KEYBLOB_RAW_FORMAT_NONE;
    context.getInt(CSSM_ATTRIBUTE_PUBLIC_KEY_FORMAT, format);
    ctx.add(CSSM_ATTRIBUTE_PUBLIC_KEY_FORMAT, format);
    CssmData *data = NULL;
    ctx(CSSM_APPLECSP_PUBKEY, (const void *)NULL, &data);
    pubKeyData = *data;
    Server::csp().allocator().free(data);
}

//
// Default getKey/getHeader calls - should never be called
//
void LocalKey::getKey()
{
	assert(false);
}

void LocalKey::getHeader(CssmKey::Header &)
{
	assert(false);
}


//
// Form a KeySpec with checking and masking
//
LocalKey::KeySpec::KeySpec(CSSM_KEYUSE usage, CSSM_KEYATTR_FLAGS attrs)
	: CssmClient::KeySpec(usage, (attrs & ~managedAttributes) | forcedAttributes)
{
	if (attrs & generatedAttributes)
		CssmError::throwMe(CSSMERR_CSP_INVALID_KEYATTR_MASK);
}

LocalKey::KeySpec::KeySpec(CSSM_KEYUSE usage, CSSM_KEYATTR_FLAGS attrs, const CssmData &label)
	: CssmClient::KeySpec(usage, (attrs & ~managedAttributes) | forcedAttributes, label)
{
	if (attrs & generatedAttributes)
		CssmError::throwMe(CSSMERR_CSP_INVALID_KEYATTR_MASK);
}
