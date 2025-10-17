/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
/* Copyright (c) 1999 Apple Computer, Inc.  All rights reserved.
 * Copyright (c) 1994-1996 NeXT Software, Inc.  All rights reserved.
 */

#ifndef _IOKIT_IOCONDITIONLOCK_H
#define _IOKIT_IOCONDITIONLOCK_H

#include <libkern/c++/OSPtr.h>
#include <libkern/c++/OSObject.h>
#include <IOKit/IOLib.h>
#include <IOKit/system.h>

#include <IOKit/IOLocks.h>

class IOConditionLock : public OSObject
{
	OSDeclareDefaultStructors(IOConditionLock);

private:
	IOLock *            cond_interlock;     // condition var Simple lock
	volatile int        condition;

	IOLock *            sleep_interlock;    // sleep lock Simple lock
	unsigned char       interruptible;
	volatile bool       want_lock;
	volatile bool       waiting;

public:
	static OSPtr<IOConditionLock> withCondition(int condition, bool inIntr = true);
	virtual bool initWithCondition(int condition, bool inIntr = true);
	virtual void free() APPLE_KEXT_OVERRIDE;

	virtual bool tryLock(); // acquire lock, no waiting
	virtual int  lock();    // acquire lock (enter critical section)
	virtual void unlock();  // release lock (leave critical section)

	virtual bool getInterruptible() const;
	virtual int  getCondition() const;
	virtual int  setCondition(int condition);

	virtual int  lockWhen(int condition);// acquire lock when condition
	virtual void unlockWith(int condition); // set condition & release lock
};

#endif /* _IOKIT_IOCONDITIONLOCK_H */
