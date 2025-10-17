/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
#define IOKIT_ENABLE_SHARED_PTR

#include <IOKit/IOFilterInterruptEventSource.h>
#include <IOKit/IOService.h>
#include <IOKit/IOKitDebug.h>
#include <IOKit/IOTimeStamp.h>
#include <IOKit/IOWorkLoop.h>
#include <IOKit/IOInterruptAccountingPrivate.h>
#include <libkern/Block.h>

#if IOKITSTATS

#define IOStatisticsInitializeCounter() \
do { \
	IOStatistics::setCounterType(IOEventSource::reserved->counter, kIOStatisticsFilterInterruptEventSourceCounter); \
} while (0)

#define IOStatisticsInterrupt() \
do { \
	IOStatistics::countInterrupt(IOEventSource::reserved->counter); \
} while (0)

#else

#define IOStatisticsInitializeCounter()
#define IOStatisticsInterrupt()

#endif /* IOKITSTATS */

#define super IOInterruptEventSource

OSDefineMetaClassAndStructors
(IOFilterInterruptEventSource, IOInterruptEventSource)
OSMetaClassDefineReservedUnused(IOFilterInterruptEventSource, 0);
OSMetaClassDefineReservedUnused(IOFilterInterruptEventSource, 1);
OSMetaClassDefineReservedUnused(IOFilterInterruptEventSource, 2);
OSMetaClassDefineReservedUnused(IOFilterInterruptEventSource, 3);
OSMetaClassDefineReservedUnused(IOFilterInterruptEventSource, 4);
OSMetaClassDefineReservedUnused(IOFilterInterruptEventSource, 5);
OSMetaClassDefineReservedUnused(IOFilterInterruptEventSource, 6);
OSMetaClassDefineReservedUnused(IOFilterInterruptEventSource, 7);

/*
 * Implement the call throughs for the private protection conversion
 */
bool
IOFilterInterruptEventSource::init(OSObject *inOwner,
    Action inAction,
    IOService *inProvider,
    int inIntIndex)
{
	return false;
}

OSSharedPtr<IOInterruptEventSource>
IOFilterInterruptEventSource::interruptEventSource(OSObject *inOwner,
    Action inAction,
    IOService *inProvider,
    int inIntIndex)
{
	return NULL;
}

bool
IOFilterInterruptEventSource::init(OSObject *inOwner,
    Action inAction,
    Filter inFilterAction,
    IOService *inProvider,
    int inIntIndex)
{
	if (!super::init(inOwner, inAction, inProvider, inIntIndex)) {
		return false;
	}

	if (!inFilterAction) {
		return false;
	}

	filterAction = inFilterAction;

	IOStatisticsInitializeCounter();

	return true;
}

OSSharedPtr<IOFilterInterruptEventSource>
IOFilterInterruptEventSource
::filterInterruptEventSource(OSObject *inOwner,
    Action inAction,
    Filter inFilterAction,
    IOService *inProvider,
    int inIntIndex)
{
	OSSharedPtr<IOFilterInterruptEventSource> me = OSMakeShared<IOFilterInterruptEventSource>();

	if (me
	    && !me->init(inOwner, inAction, inFilterAction, inProvider, inIntIndex)) {
		return nullptr;
	}

	return me;
}


OSSharedPtr<IOFilterInterruptEventSource>
IOFilterInterruptEventSource
::filterInterruptEventSource(OSObject *inOwner,
    IOService *inProvider,
    int inIntIndex,
    ActionBlock inAction,
    FilterBlock inFilterAction)
{
	OSSharedPtr<IOFilterInterruptEventSource> me = OSMakeShared<IOFilterInterruptEventSource>();

	FilterBlock filter = Block_copy(inFilterAction);
	if (!filter) {
		return nullptr;
	}

	if (me
	    && !me->init(inOwner, (Action) NULL, (Filter) (void (*)(void))filter, inProvider, inIntIndex)) {
		Block_release(filter);
		return nullptr;
	}
	me->flags |= kFilterBlock;
	me->setActionBlock((IOEventSource::ActionBlock) inAction);

	return me;
}


void
IOFilterInterruptEventSource::free( void )
{
	if ((kFilterBlock & flags) && filterActionBlock) {
		Block_release(filterActionBlock);
	}

	super::free();
}

void
IOFilterInterruptEventSource::signalInterrupt()
{
	bool trace = (gIOKitTrace & kIOTraceIntEventSource) ? true : false;

	IOStatisticsInterrupt();
	producerCount++;

	if (trace) {
		IOTimeStampStartConstant(IODBG_INTES(IOINTES_SEMA), VM_KERNEL_ADDRHIDE(this), VM_KERNEL_ADDRHIDE(owner));
	}

	signalWorkAvailable();

	if (trace) {
		IOTimeStampEndConstant(IODBG_INTES(IOINTES_SEMA), VM_KERNEL_ADDRHIDE(this), VM_KERNEL_ADDRHIDE(owner));
	}
}


IOFilterInterruptEventSource::Filter
IOFilterInterruptEventSource::getFilterAction() const
{
	if (kFilterBlock & flags) {
		return NULL;
	}
	return filterAction;
}

IOFilterInterruptEventSource::FilterBlock
IOFilterInterruptEventSource::getFilterActionBlock() const
{
	if (kFilterBlock & flags) {
		return filterActionBlock;
	}
	return NULL;
}

void
IOFilterInterruptEventSource::normalInterruptOccurred
(void */*refcon*/, IOService */*prov*/, int /*source*/)
{
	bool        filterRes;
	uint64_t    startTime = 0;
	uint64_t    endTime = 0;
	bool    trace = (gIOKitTrace & kIOTraceIntEventSource) ? true : false;

	if (trace) {
		IOTimeStampStartConstant(IODBG_INTES(IOINTES_FILTER),
		    VM_KERNEL_UNSLIDE(filterAction), VM_KERNEL_ADDRHIDE(owner), VM_KERNEL_ADDRHIDE(this), VM_KERNEL_ADDRHIDE(workLoop));
	}

	if (IOInterruptEventSource::reserved->statistics) {
		if (IA_GET_STATISTIC_ENABLED(kInterruptAccountingFirstLevelTimeIndex)
		    || IOInterruptEventSource::reserved->statistics->enablePrimaryTimestamp) {
			startTime = mach_absolute_time();
		}
		if (IOInterruptEventSource::reserved->statistics->enablePrimaryTimestamp) {
			IOInterruptEventSource::reserved->statistics->primaryTimestamp = startTime;
		}
	}

	// Call the filter.
	if (kFilterBlock & flags) {
		filterRes = (filterActionBlock)(this);
	} else {
		filterRes = (*filterAction)(owner, this);
	}

	if (IOInterruptEventSource::reserved->statistics) {
		if (IA_GET_STATISTIC_ENABLED(kInterruptAccountingFirstLevelCountIndex)) {
			IA_ADD_VALUE(&IOInterruptEventSource::reserved->statistics->interruptStatistics[kInterruptAccountingFirstLevelCountIndex], 1);
		}

		if (IA_GET_STATISTIC_ENABLED(kInterruptAccountingFirstLevelTimeIndex)) {
			endTime = mach_absolute_time();
			IA_ADD_VALUE(&IOInterruptEventSource::reserved->statistics->interruptStatistics[kInterruptAccountingFirstLevelTimeIndex], endTime - startTime);
		}
	}

	if (trace) {
		IOTimeStampEndConstant(IODBG_INTES(IOINTES_FILTER),
		    VM_KERNEL_ADDRHIDE(filterAction), VM_KERNEL_ADDRHIDE(owner),
		    VM_KERNEL_ADDRHIDE(this), VM_KERNEL_ADDRHIDE(workLoop));
	}

	if (filterRes) {
		signalInterrupt();
	}
}

void
IOFilterInterruptEventSource::disableInterruptOccurred
(void */*refcon*/, IOService *prov, int source)
{
	bool        filterRes;
	uint64_t    startTime = 0;
	uint64_t    endTime = 0;
	bool    trace = (gIOKitTrace & kIOTraceIntEventSource) ? true : false;

	if (trace) {
		IOTimeStampStartConstant(IODBG_INTES(IOINTES_FILTER),
		    VM_KERNEL_UNSLIDE(filterAction), VM_KERNEL_ADDRHIDE(owner), VM_KERNEL_ADDRHIDE(this), VM_KERNEL_ADDRHIDE(workLoop));
	}

	if (IOInterruptEventSource::reserved->statistics) {
		if (IA_GET_STATISTIC_ENABLED(kInterruptAccountingFirstLevelTimeIndex)
		    || IOInterruptEventSource::reserved->statistics->enablePrimaryTimestamp) {
			startTime = mach_absolute_time();
		}
		if (IOInterruptEventSource::reserved->statistics->enablePrimaryTimestamp) {
			IOInterruptEventSource::reserved->statistics->primaryTimestamp = startTime;
		}
	}

	// Call the filter.
	if (kFilterBlock & flags) {
		filterRes = (filterActionBlock)(this);
	} else {
		filterRes = (*filterAction)(owner, this);
	}

	if (IOInterruptEventSource::reserved->statistics) {
		if (IA_GET_STATISTIC_ENABLED(kInterruptAccountingFirstLevelCountIndex)) {
			IA_ADD_VALUE(&IOInterruptEventSource::reserved->statistics->interruptStatistics[kInterruptAccountingFirstLevelCountIndex], 1);
		}

		if (IA_GET_STATISTIC_ENABLED(kInterruptAccountingFirstLevelTimeIndex)) {
			endTime = mach_absolute_time();
			IA_ADD_VALUE(&IOInterruptEventSource::reserved->statistics->interruptStatistics[kInterruptAccountingFirstLevelTimeIndex], endTime - startTime);
		}
	}

	if (trace) {
		IOTimeStampEndConstant(IODBG_INTES(IOINTES_FILTER),
		    VM_KERNEL_UNSLIDE(filterAction), VM_KERNEL_ADDRHIDE(owner), VM_KERNEL_ADDRHIDE(this), VM_KERNEL_ADDRHIDE(workLoop));
	}

	if (filterRes) {
		prov->disableInterrupt(source); /* disable the interrupt */
		signalInterrupt();
	}
}
