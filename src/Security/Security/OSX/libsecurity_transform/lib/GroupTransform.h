/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 7, 2022.
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

#ifndef __GROUP_TRANSFORM__
#define __GROUP_TRANSFORM__


#include "Transform.h"
#include "TransformFactory.h"

extern CFStringRef kSecGroupTransformType;

class GroupTransform : public Transform
{
protected:
	std::string DebugDescription();
	virtual void FinalizePhase2();
    virtual bool validConnectionPoint(CFStringRef attributeName);
	GroupTransform();
	CFMutableArrayRef mMembers;
	dispatch_group_t mAllChildrenFinalized;
    dispatch_group_t mPendingStartupActivity;

    void RecurseForAllNodes(dispatch_group_t group, CFErrorRef *errorOut, bool parallel, bool opExecutesOnGroups, Transform::TransformOperation op);
    
public:
	virtual ~GroupTransform();

	static CFTypeRef Make();
	static TransformFactory* MakeTransformFactory();
	
	static CFTypeID GetCFTypeID();
	
	void AddMemberToGroup(SecTransformRef member);
	void RemoveMemberFromGroup(SecTransformRef member);
	bool HasMember(SecTransformRef member);
	
	void AddAllChildrenFinalizedCallback(dispatch_queue_t run_on, dispatch_block_t callback);
	void ChildStartedFinalization(Transform *child);

	SecTransformRef FindFirstTransform();		// defined as the transform to which input is attached
	SecTransformRef FindLastTransform();		// defined as the transform to which the monitor is attached
	SecTransformRef FindMonitor();
	SecTransformRef GetAnyMember();
	
	SecTransformRef FindByName(CFStringRef name);
    
    // A group should delay destruction while excution is starting
    void StartingExecutionInGroup();
    void StartedExecutionInGroup(bool successful);
	
	virtual CFDictionaryRef Externalize(CFErrorRef* error);
	
    CFErrorRef ForAllNodes(bool parallel, bool opExecutesOnGroups, Transform::TransformOperation op);
	void ForAllNodesAsync(bool opExecutesOnGroups, dispatch_group_t group, Transform::TransformAsyncOperation op);

    CFStringRef DotForDebugging();
};



#endif
