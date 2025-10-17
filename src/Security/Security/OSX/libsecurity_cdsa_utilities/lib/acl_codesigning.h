/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
// acl_codesigning - ACL subject for signature of calling application
//
// Note:
// Once upon a time, a code signature was a single binary blob, a "signature".
// Then we added an optional second blob, a "comment". The comment was only
// ancilliary (non-security) data first, but then we added more security data
// to it later. Now, the security-relevant data is kept in a (signature, comment)
// pair, all of which is relevant for the security of such subjects.
// Don't read any particular semantics into this separation. It is historical only
// (having to do with backward binary compatibility of ACL blobs).
//
#ifndef _H_ACL_CODESIGNING
#define _H_ACL_CODESIGNING

#include <security_cdsa_utilities/cssmdata.h>
#include <security_cdsa_utilities/cssmacl.h>
#include <security_cdsa_utilities/osxverifier.h>

namespace Security {


//
// The CodeSignature subject type matches a code signature applied to the
// disk image that originated the client process.
//
class CodeSignatureAclSubject : public AclSubject, public OSXVerifier {
public:
	class Maker; friend class Maker;
	
	static const size_t commentBagAlignment = 4;
	
    CodeSignatureAclSubject(const SHA1::Byte *hash, const std::string &path)
		: AclSubject(CSSM_ACL_SUBJECT_TYPE_CODE_SIGNATURE), OSXVerifier(hash, path) { }

	CodeSignatureAclSubject(const OSXVerifier &verifier)
		: AclSubject(CSSM_ACL_SUBJECT_TYPE_CODE_SIGNATURE), OSXVerifier(verifier) { }
	
    bool validates(const AclValidationContext &baseCtx) const;
    CssmList toList(Allocator &alloc) const;
    
    void exportBlob(Writer::Counter &pub, Writer::Counter &priv);
    void exportBlob(Writer &pub, Writer &priv);
	
	IFDUMP(void debugDump() const);
    virtual CFStringRef createACLDebugString() const;

public:
    class Environment : public virtual AclValidationEnvironment {
    public:
		virtual bool verifyCodeSignature(const OSXVerifier &verifier,
			const AclValidationContext &context) = 0;
    };

public:
    class Maker : public AclSubject::Maker {
    public:
    	Maker()
		: AclSubject::Maker(CSSM_ACL_SUBJECT_TYPE_CODE_SIGNATURE) { }
    	CodeSignatureAclSubject *make(const TypedList &list) const;
    	CodeSignatureAclSubject *make(Version version, Reader &pub, Reader &priv) const;
	
	private:
		CodeSignatureAclSubject *make(const SHA1::Byte *hash, const CssmData &commentBag) const;
    };
};

} // end namespace Security



#endif //_H_ACL_CODESIGNING
