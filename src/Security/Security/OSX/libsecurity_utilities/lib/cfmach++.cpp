/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 13, 2025.
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
#include <security_utilities/cfmach++.h>


namespace Security {
namespace MachPlusPlus {


//
// Construct CFAutoPorts
//
CFAutoPort::CFAutoPort()
	: mEnabled(false)
{ }

CFAutoPort::CFAutoPort(mach_port_t p)
	: Port(p), mEnabled(false)
{ }


//
// On destruction, make sure we're disengaged from the CFRunLoop
//
CFAutoPort::~CFAutoPort()
{
	disable();
	
	// invalidate everything
	if (mPort)
	{
		CFMachPortInvalidate(mPort);
		CFRunLoopSourceInvalidate(mSource);
	}
}


//
// enable() will lazily allocate needed resources, then click into the runloop
//
void CFAutoPort::enable()
{
	if (!mEnabled) {
		if (!*this)
			allocate();
		if (!mPort) {
			// first-time creation of CF resources
			CFMachPortContext ctx = { 1, this, NULL, NULL, NULL };
			CFMachPortRef machPort = CFMachPortCreateWithPort(NULL, port(), cfCallback, &ctx, NULL);
			if (machPort != NULL)
			{
				// using take here because "assignment" causes an extra retain, which will make the
				// CF objects leak when this data structure goes away.
				mPort.take(machPort);
				
				CFRunLoopSourceRef sr = CFMachPortCreateRunLoopSource(NULL, mPort, 10);
				mSource.take(sr);
			}
			if (!mPort || !mSource)
				CFError::throwMe();		// CF won't tell us why...
		}
		CFRunLoopAddSource(CFRunLoopGetCurrent(), mSource, kCFRunLoopCommonModes);
		mEnabled = true;
		secinfo("autoport", "%p enabled", this);
	}
}


//
// Disable() just removes us from the runloop. All the other resources stay
// around, ready to be re-enable()d.
//
void CFAutoPort::disable()
{
	if (mEnabled) {
		CFRunLoopRemoveSource(CFRunLoopGetCurrent(), mSource, kCFRunLoopCommonModes);
		mEnabled = false;
		secinfo("autoport", "%p disabled", this);
	}
}


//
// The CF-sponsored port callback.
// We pass this to our receive() virtual and eat all exceptions.
//
static int gNumTimesCalled = 0;

void CFAutoPort::cfCallback(CFMachPortRef cfPort, void *msg, CFIndex size, void *context)
{
	++gNumTimesCalled;
	secinfo("adhoc", "Callback was called %d times.", gNumTimesCalled);

	Message message(msg, (mach_msg_size_t)size);
	try {
		reinterpret_cast<CFAutoPort *>(context)->receive(message);
	} catch (...) {
		secinfo("autoport", "%p receive handler failed with exception", context);
	}
}


} // end namespace MachPlusPlus
} // end namespace Security
