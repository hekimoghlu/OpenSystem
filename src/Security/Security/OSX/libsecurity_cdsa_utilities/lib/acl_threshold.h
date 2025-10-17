/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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
// acl_threshold - Threshold-based group ACL subjects.
//
// This subject type implements threshold (k of n) subjects as per CSSM standard.
// Subsubjects are stored and evaluated in the order received. Any subsubject
// is presented with all subsamples of the corresponding threshold sample, but
// not any other samples possibly present in the credentials. Subsubject evaluation
// stops as soon as the threshold is satisfied, or as soon as it becomes numerically
// impossible to satisfy the threshold with future matches.
// Note that this subject will reject out of hand any threshold sample that
// contains more than <n> subsamples. This defeats "sample stuffing" attacks
// where the attacker provides thousands of samples in the hope that some may
// match by accident. It will however accept threshold samples with fewer than
// <n> subsamples, as long as there are at least <k> subsamples.
//
#ifndef _ACL_THRESHOLD
#define _ACL_THRESHOLD

#include <security_cdsa_utilities/cssmacl.h>
#include <vector>


namespace Security {

class ThresholdAclSubject : public SimpleAclSubject {
    typedef ObjectAcl::AclSubjectPointer AclSubjectPointer;
    typedef vector<AclSubjectPointer> AclSubjectVector;
public:
    bool validates(const AclValidationContext &baseCtx, const TypedList &sample) const;
    CssmList toList(Allocator &alloc) const;
    
    ThresholdAclSubject(uint32 n, uint32 k, const AclSubjectVector &subSubjects);
    
    void exportBlob(Writer::Counter &pub, Writer::Counter &priv);
    void exportBlob(Writer &pub, Writer &priv);
	
	unsigned count() const { return totalSubjects; }
	AclSubject *subject(unsigned n) const { return elements[n]; }
	void add(AclSubject *subject, unsigned beforePosition);
	
	IFDUMP(void debugDump() const);
    virtual CFStringRef createACLDebugString() const;

    class Maker : public AclSubject::Maker {
    public:
    	Maker() : AclSubject::Maker(CSSM_ACL_SUBJECT_TYPE_THRESHOLD) { }
    	ThresholdAclSubject *make(const TypedList &list) const;
    	ThresholdAclSubject *make(Version, Reader &pub, Reader &priv) const;
    };
    
private:
    uint32 minimumNeeded;				// number of matches needed
    uint32 totalSubjects;				// number of subSubjects
    AclSubjectVector elements;			// sub-subject vector

    template <class Action>
    void exportBlobForm(Action &pub, Action &priv);
};

} // namespace Security


#endif //_ACL_THRESHOLD
