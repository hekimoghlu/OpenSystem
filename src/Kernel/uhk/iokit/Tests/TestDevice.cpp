/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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
#if DEBUG

#include "Tests.h"

#include <IOKit/IOCommandQueue.h>
#include <IOKit/IOInterruptEventSource.h>
#include <IOKit/IOWorkLoop.h>

#include <mach/sync_policy.h>

#define super OSObject

static TestDevice *sDevice;

static mach_timespec_t hundredMill = { 0, 100000000 };
static semaphore_port_t completeSema;

OSDefineMetaClassAndStructors(TestDevice, OSObject)

kern_return_t
TestDevice::enqueueCommand(bool sleep,
    TestDeviceAction act, int tag, void *dataP)
{
	return commQ->enqueueCommand(sleep, (void *) act, (void *) tag, dataP);
}

bool
TestDevice::init()
{
	if (!super::init()) {
		return false;
	}

	workLoop = IOWorkLoop::workLoop();
	if (!workLoop) {
		return false;
	}

	commQ = IOCommandQueue::commandQueue
	    (this, (IOCommandQueueAction) & rawCommandOccurred, 8);
	if (!commQ || kIOReturnSuccess != workLoop->addEventSource(commQ)) {
		return false;
	}

	intES = IOInterruptEventSource::interruptEventSource
	    (this, (IOInterruptEventAction) & interruptAction);
	if (!intES || kIOReturnSuccess != workLoop->addEventSource(intES)) {
		return false;
	}

	return true;
}

void
TestDevice::free()
{
	if (intES) {
		intES->release();
	}
	if (commQ) {
		commQ->release();
	}
	if (workLoop) {
		workLoop->release();
	}

	super::free();
}

void
TestDevice::rawCommandOccurred
(void *field0, void *field1, void *field2, void *)
{
	(*(TestDeviceAction) field0)(this, (int) field1, field2);
}

void
TestDevice::interruptAction(IOInterruptEventSource *, int count)
{
	logPrintf(("I(%d, %d) ", count, ++intCount));
}

void
TestDevice::producer1Action(int tag)
{
	logPrintf(("C1(%d) ", tag));
}

void
TestDevice::producer2Action(int tag, void *count)
{
	logPrintf(("C2(%d,%d) ", tag, (int) count));
	if (!(tag % 10)) {
		IOSleep(1000);
	}
}

void
TestDevice::alarm()
{
	intES->interruptOccurred(0, 0, 0);
	IOScheduleFunc((IOThreadFunc) alarm, (void *) this, hundredMill, 1);
}

static void
producer(void *inProducerId)
{
	int producerId = (int) inProducerId;
	TestDeviceAction command;
	int i;

	semaphore_wait(completeSema);

	if (producerId & 1) {
		command = (TestDeviceAction) sDevice->producer1Action;
	} else {
		command = (TestDeviceAction) sDevice->producer2Action;
	}

	for (i = 0; i < 5 * (producerId << 1); i++) {
		sDevice->enqueueCommand
		(true, command, i, (void *) (i % (producerId + 1)));
		if (!(i % (producerId + 1))) {
			/* cthread_yield() */;
		}
		logPrintf(("TestDevice(%d): %d\n", producerId, i));
	}

	logPrintf(("TestDevice: producer %d exiting\n", producerId));
	semaphore_signal(completeSema);

	IOExitThread(producerId);
}

void
testWorkLoop()
{
	int i;

	sDevice = new TestDevice;
	if (!sDevice || !sDevice->init()) {
		if (sDevice) {
			sDevice->free();
		}
		logPrintf(("TestDevice: couldn't create device instance\n"));
		return;
	}

	IOSleep(1000);

	IOScheduleFunc((IOThreadFunc) sDevice->alarm, sDevice, hundredMill, 1);

	IOSleep(2000);

	if (KERN_SUCCESS
	    != semaphore_create(kernel_task, &completeSema, SYNC_POLICY_FIFO, 4)) {
		return;
	}

	IOCreateThread(producer, (void *) 4);
	IOCreateThread(producer, (void *) 3);
	IOCreateThread(producer, (void *) 2);
	IOCreateThread(producer, (void *) 1);

	IOSleep(2000);

	for (i = 0; i < 4; i++) {
		semaphore_wait(completeSema);
	}

	IOUnscheduleFunc((IOThreadFunc) sDevice->alarm, sDevice);

	sDevice->free(); sDevice = 0;

	logPrintf(("TestDevice: exiting\n"));
}

#endif /* DEBUG */
