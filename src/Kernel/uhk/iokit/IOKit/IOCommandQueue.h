/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
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
/*
 *  Copyright (c) 1998 Apple Computer, Inc.  All rights reserved.
 *
 *  HISTORY
 *   1998-7-13	Godfrey van der Linden(gvdl)
 *       Created.
 *   1998-10-30	Godfrey van der Linden(gvdl)
 *       Converted to C++
 *   1999-9-22	Godfrey van der Linden(gvdl)
 *       Deprecated
 *  ]*/
#ifndef _IOKIT_IOCOMMANDQUEUE_H
#define _IOKIT_IOCOMMANDQUEUE_H

#include <IOKit/IOEventSource.h>
#include <libkern/c++/OSPtr.h>

class IOCommandQueue;

typedef void (*IOCommandQueueAction)
(OSObject *, void *field0, void *field1, void *field2, void *field3);

class IOCommandQueue : public IOEventSource
{
	OSDeclareDefaultStructors(IOCommandQueue);

protected:
	static const int kIOCQDefaultSize = 128;

	void *queue;
	IOLock *producerLock;
	semaphore_port_t producerSema;
	int producerIndex, consumerIndex;
	int size;

	virtual void free() APPLE_KEXT_OVERRIDE;

	virtual bool checkForWork() APPLE_KEXT_OVERRIDE;

public:
	static OSPtr<IOCommandQueue> commandQueue(OSObject *inOwner,
	    IOCommandQueueAction inAction = NULL,
	    int inSize = kIOCQDefaultSize)
	APPLE_KEXT_DEPRECATED;
	virtual bool init(OSObject *inOwner,
	    IOCommandQueueAction inAction = NULL,
	    int inSize = kIOCQDefaultSize)
	APPLE_KEXT_DEPRECATED;

	virtual kern_return_t enqueueCommand(bool gotoSleep = true,
	    void *field0 = NULL, void *field1 = NULL,
	    void *field2 = NULL, void *field3 = NULL)
	APPLE_KEXT_DEPRECATED;

// WARNING:  This function can only be safely called from the appropriate
// work loop context.  You should check IOWorkLoop::onThread is true.
//
// For each entry in the commandQueue call the target/action.
// Lockout all new entries to the queue while iterating.
// If the input fields are zero then the queue's owner/action will be used.
	virtual int performAndFlush(OSObject *target = NULL,
	    IOCommandQueueAction inAction = NULL)
	APPLE_KEXT_DEPRECATED;
};

#endif /* !_IOKIT_IOCOMMANDQUEUE_H */
