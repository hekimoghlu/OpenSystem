/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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
// child - track a single child process and its belongings
//
#ifndef _CHILD_H_
#define _CHILD_H_  1

#include <security_utilities/mach++.h>
#include <security_utilities/unixchild.h>

using MachPlusPlus::Port;
using MachPlusPlus::TaskPort;


//
// ServerChild builds on the generic UNIX Child abstraction.
// The child is expected to engage in a checkin protocol after launch,
// whereby it RPC-calls childCheckIn in securityd and thus authenticates
// and declares readiness to provide service.
//
// @@@ PerWhat are these, if they are at all?
//
class ServerChild : public UnixPlusPlus::Child {
public:
	ServerChild();
	~ServerChild();
	
	Port servicePort() const { return mServicePort; }
	bool ready() const { return mServicePort; }

public:
	static void checkIn(Port servicePort, pid_t pid);

protected:
	void childAction() = 0;		// must be provided by subclass
	void parentAction();		// fully implemented
	void dying();				// fully implemented

private:
	Port mServicePort;			// child's main service port

private:
	typedef map<pid_t, ServerChild *> CheckinMap;
	static CheckinMap mCheckinMap;

	// The parent side will wait on mCheckinCond until the child checks in
	// or fails. During that time ONLY, mCheckinLock protects the entire Child
	// object.
	static Mutex mCheckinLock;
	Condition mCheckinCond;
};


#endif // _CHILD_H_
