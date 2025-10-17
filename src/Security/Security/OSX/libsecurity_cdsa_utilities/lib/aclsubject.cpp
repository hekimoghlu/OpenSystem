/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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
// aclsubject - abstract ACL subject implementation
//
#include <security_cdsa_utilities/cssmacl.h>
#include <security_cdsa_utilities/cssmbridge.h>
#include <security_utilities/endian.h>
#include <security_utilities/debugging.h>
#include <algorithm>
#include <cstdarg>


//
// Validation contexts
//
AclValidationContext::~AclValidationContext()
{ /* virtual */ }


void AclValidationContext::init(ObjectAcl *acl, AclSubject *subject)
{
	mAcl = acl;
	mSubject = subject;
}


const char *AclValidationContext::credTag() const
{
	return mCred ? mCred->tag() : NULL;
}

std::string AclValidationContext::s_credTag() const
{
	const char *s = this->credTag();
	return s ? s : "";
}

const char *AclValidationContext::entryTag() const
{
	return mEntryTag;
}

void AclValidationContext::entryTag(const char *tag)
{
	mEntryTag = (tag && tag[0]) ? tag : NULL;
}

void AclValidationContext::entryTag(const std::string &tag)
{
	mEntryTag = tag.empty() ? NULL : tag.c_str();
}


//
// Common (basic) features of AclSubjects
//
AclSubject::AclSubject(uint32 type, Version v /* = 0 */)
	: mType(type), mVersion(v)
{
	assert(!(type & versionMask));
}

AclSubject::~AclSubject()
{ }

AclValidationEnvironment::~AclValidationEnvironment()
{ }

Adornable &AclValidationEnvironment::store(const AclSubject *subject)
{
	CssmError::throwMe(CSSM_ERRCODE_ACL_SUBJECT_TYPE_NOT_SUPPORTED);
}

void AclSubject::exportBlob(Writer::Counter &, Writer::Counter &)
{ }

void AclSubject::exportBlob(Writer &, Writer &)
{ }

void AclSubject::importBlob(Reader &, Reader &)
{ }

void AclSubject::reset()
{ }

AclSubject::Maker::~Maker()
{
}


//
// A SimpleAclSubject accepts only a single type of sample, validates
// samples independently, and makes no use of certificates.
//
bool SimpleAclSubject::validates(const AclValidationContext &ctx) const
{
    for (uint32 n = 0; n < ctx.count(); n++) {
        const TypedList &sample = ctx[n];
        if (!sample.isProper())
            CssmError::throwMe(CSSM_ERRCODE_INVALID_SAMPLE_VALUE);
        if (sample.type() == type() && validates(ctx, sample)) {
			ctx.matched(ctx[n]);
            return true;	// matched this sample; validation successful
		}
    }
    return false;
}

CFStringRef SimpleAclSubject::createACLDebugString() const
{
    return CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("<SimpleAclSubject(type:%d)>"), this->type());
}

//
// AclSubjects always have a (virtual) dump method.
// It's empty unless DEBUGDUMP is enabled.
//
void AclSubject::debugDump() const
{
#if defined(DEBUGDUMP)
	switch (type()) {
	case CSSM_ACL_SUBJECT_TYPE_ANY:
		Debug::dump("ANY");
		break;
	default:
		Debug::dump("subject type=%d", type());
		break;
	}
#endif //DEBUGDUMP
}

#if defined(DEBUGDUMP)

void AclSubject::dump(const char *title) const
{
	Debug::dump(" ** %s ", title);
	this->debugDump();
	Debug::dump("\n");
}

#endif //DEBUGDUMP

CFStringRef AclValidationContext::createACLDebugString() const
{
    // An ACL Validation context doesn't really exist until init() has been called on it. Check the internal pointers against the known pointer pattern.
    CFStringRef objectDesc = NULL;
    CFStringRef subjectDesc = NULL;

    if(mAcl == NULL) {
        objectDesc = CFSTR("null");
    } else if(mAcl == (void*)0xDEADDEADDEADDEAD) {
        objectDesc = CFSTR("not-init");
    } else {
        objectDesc = mAcl->createACLDebugString();
    }

    if(mSubject == NULL) {
        subjectDesc = CFSTR("null");
    } else if(mSubject == (void*)0xDEADDEADDEADDEAD) {
        subjectDesc = CFSTR("not-init");
    } else {
        subjectDesc = mSubject->createACLDebugString();
    }

    CFStringRef s = CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("<AclValidationContext(action:%d)SUBJECT[%@]OBJECT[%@]>"),
                                             mAuth,
                                             subjectDesc,
                                             objectDesc);

    if(objectDesc) {
        CFRelease(objectDesc);
    }
    if(subjectDesc) {
        CFRelease(subjectDesc);
    }
    return s;
}

CFStringRef AclSubject::createACLDebugString() const
{
    switch (type()) {
        case CSSM_ACL_SUBJECT_TYPE_ANY:
            return CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("<AclSubject[type:ANY]>"));

        default:
            return CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("<AclSubject[type:%d]>"), mType);
    }
}
