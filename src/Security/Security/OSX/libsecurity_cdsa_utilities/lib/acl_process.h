/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
// acl_process - Process-attribute ACL subject type.
//
// NOTE:
// The default Environment provides data about the current process (the one that
// validate() is run in). If this isn't right for you (e.g. because you want to
// validate against a process on the other side of some IPC connection), you must
// make your own version of Environment and pass it to validate().
//
#ifndef _ACL_PROCESS
#define _ACL_PROCESS

#include <security_cdsa_utilities/cssmacl.h>
#include <string>

namespace Security
{

class AclProcessSubjectSelector
    : public PodWrapper<AclProcessSubjectSelector, CSSM_ACL_PROCESS_SUBJECT_SELECTOR> {
public:
    AclProcessSubjectSelector()
    { version = CSSM_ACL_PROCESS_SELECTOR_CURRENT_VERSION; mask = 0; }
    
    bool uses(uint32 m) const { return mask & m; }
};


//
// The ProcessAclSubject matches process attributes securely identified
// by the system across IPC channels.
//
class ProcessAclSubject : public AclSubject {
public:
    bool validates(const AclValidationContext &baseCtx) const;
    CssmList toList(Allocator &alloc) const;

    ProcessAclSubject(const AclProcessSubjectSelector &selector)
    : AclSubject(CSSM_ACL_SUBJECT_TYPE_PROCESS),
      select(selector) { }

    void exportBlob(Writer::Counter &pub, Writer::Counter &priv);
    void exportBlob(Writer &pub, Writer &priv);
	
	IFDUMP(void debugDump() const);
    virtual CFStringRef createACLDebugString() const;

public:
    class Environment : public virtual AclValidationEnvironment {
    public:
        virtual uid_t getuid() const;	// retrieve effective userid to match
        virtual gid_t getgid() const;	// retrieve effective groupid to match
    };
    
public:
    class Maker : public AclSubject::Maker {
    public:
    	Maker() : AclSubject::Maker(CSSM_ACL_SUBJECT_TYPE_PROCESS) { }
    	ProcessAclSubject *make(const TypedList &list) const;
    	ProcessAclSubject *make(Version, Reader &pub, Reader &priv) const;
    };

private:
    AclProcessSubjectSelector select;
};

} // end namespace Security


#endif //_ACL_PROCESS
