/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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
// cfmach++ - a marriage of CoreFoundation with Mach/C++
//
#ifndef _H_CFMACHPP
#define _H_CFMACHPP

#include <security_utilities/mach++.h>
#include <security_utilities/cfutilities.h>
#include <CoreFoundation/CFMachPort.h>
#include <CoreFoundation/CFRunLoop.h>


namespace Security {
namespace MachPlusPlus {


//
// A self-receiving MachPlusPlus::Message.
// Data is delivered through the standard CFRunLoop of the current thread.
// Note that CFAutoPort does NOT own the Port; you must release it yourself
// if you're done with it.
//
class CFAutoPort : public Port {
public:
	CFAutoPort();	// lazily allocates port later
	CFAutoPort(mach_port_t port); // use this port (must have receive right)
	virtual ~CFAutoPort();
	
	void enable();
	void disable();
	
	virtual void receive(const Message &msg) = 0;
	
private:
	CFRef<CFMachPortRef> mPort;
	CFRef<CFRunLoopSourceRef> mSource;
	bool mEnabled;
	
	static void cfCallback(CFMachPortRef cfPort, void *msg, CFIndex size, void *context);
};



} // end namespace MachPlusPlus
} // end namespace Security

#endif //_H_MACHPP
