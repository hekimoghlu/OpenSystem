/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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

#include <IOKit/IOInterruptAccountingPrivate.h>
#include <IOKit/IOKernelReporters.h>

uint32_t gInterruptAccountingStatisticBitmask =
#if !defined(__arm__)
    /* Disable timestamps for older ARM platforms; they are expensive. */
    IA_GET_ENABLE_BIT(kInterruptAccountingFirstLevelTimeIndex) |
    IA_GET_ENABLE_BIT(kInterruptAccountingSecondLevelCPUTimeIndex) |
    IA_GET_ENABLE_BIT(kInterruptAccountingSecondLevelSystemTimeIndex) |
#endif
    IA_GET_ENABLE_BIT(kInterruptAccountingFirstLevelCountIndex) |
    IA_GET_ENABLE_BIT(kInterruptAccountingSecondLevelCountIndex);

IOLock * gInterruptAccountingDataListLock = NULL;
queue_head_t gInterruptAccountingDataList;

void
interruptAccountingInit(void)
{
	int bootArgValue = 0;

	if (PE_parse_boot_argn("interrupt_accounting", &bootArgValue, sizeof(bootArgValue))) {
		gInterruptAccountingStatisticBitmask = bootArgValue;
	}

	gInterruptAccountingDataListLock = IOLockAlloc();

	assert(gInterruptAccountingDataListLock);

	queue_init(&gInterruptAccountingDataList);
}

void
interruptAccountingDataAddToList(IOInterruptAccountingData * data)
{
	IOLockLock(gInterruptAccountingDataListLock);
	queue_enter(&gInterruptAccountingDataList, data, IOInterruptAccountingData *, chain);
	IOLockUnlock(gInterruptAccountingDataListLock);
}

void
interruptAccountingDataRemoveFromList(IOInterruptAccountingData * data)
{
	IOLockLock(gInterruptAccountingDataListLock);
	queue_remove(&gInterruptAccountingDataList, data, IOInterruptAccountingData *, chain);
	IOLockUnlock(gInterruptAccountingDataListLock);
}

void
interruptAccountingDataUpdateChannels(IOInterruptAccountingData * data, IOSimpleReporter * reporter)
{
	uint64_t i = 0;

	for (i = 0; i < IA_NUM_INTERRUPT_ACCOUNTING_STATISTICS; i++) {
		if (IA_GET_STATISTIC_ENABLED(i)) {
			reporter->setValue(IA_GET_CHANNEL_ID(data->interruptIndex, i), data->interruptStatistics[i]);
		}
	}
}

void
interruptAccountingDataInheritChannels(IOInterruptAccountingData * data, IOSimpleReporter * reporter)
{
	uint64_t i = 0;

	for (i = 0; i < IA_NUM_INTERRUPT_ACCOUNTING_STATISTICS; i++) {
		if (IA_GET_STATISTIC_ENABLED(i)) {
			data->interruptStatistics[i] = reporter->getValue(IA_GET_CHANNEL_ID(data->interruptIndex, i));
		}
	}
}
