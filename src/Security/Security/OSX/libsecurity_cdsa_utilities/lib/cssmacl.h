/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
// cssmacl - core ACL management interface.
//
// This header once contain the entire canonical ACL machinery. It's been split up
// since, into objectacl.h and aclsubject.h. What remains is the PodWrapper for
// ResourceControlContext, because nobody else wants it.
//
#ifndef _CSSMACL
#define _CSSMACL

#include <security_cdsa_utilities/objectacl.h>
#include <security_cdsa_utilities/aclsubject.h>


namespace Security {


//
// This bastard child of two different data structure sets has no natural home.
// We'll take pity on it here.
//
class ResourceControlContext : public PodWrapper<ResourceControlContext, CSSM_RESOURCE_CONTROL_CONTEXT> {
public:
    ResourceControlContext() { clearPod(); }
    ResourceControlContext(const AclEntryInput &initial,
		const AccessCredentials *cred = NULL)
    { InitialAclEntry = initial; AccessCred = const_cast<AccessCredentials *>(cred); }
    
	AclEntryInput &input()		{ return AclEntryInput::overlay(InitialAclEntry); }
    operator AclEntryInput &()	{ return input(); }
    AccessCredentials *credentials() const { return AccessCredentials::overlay(AccessCred); }
	void credentials(const CSSM_ACCESS_CREDENTIALS *creds)
		{ AccessCred = const_cast<CSSM_ACCESS_CREDENTIALS *>(creds); }
};

} // end namespace Security


#endif //_CSSMACL
