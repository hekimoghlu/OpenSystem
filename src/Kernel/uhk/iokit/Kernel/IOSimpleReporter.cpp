/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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

#include <libkern/c++/OSSharedPtr.h>
#include <IOKit/IOKernelReportStructs.h>
#include <IOKit/IOKernelReporters.h>
#include "IOReporterDefs.h"

#define super IOReporter
OSDefineMetaClassAndStructors(IOSimpleReporter, IOReporter);

/* static */
OSSharedPtr<IOSimpleReporter>
IOSimpleReporter::with(IOService *reportingService,
    IOReportCategories categories,
    IOReportUnit unit)
{
	OSSharedPtr<IOSimpleReporter> reporter;

	reporter = OSMakeShared<IOSimpleReporter>();
	if (!reporter) {
		return nullptr;
	}

	if (!reporter->initWith(reportingService, categories, unit)) {
		return nullptr;
	}

	return reporter;
}

bool
IOSimpleReporter::initWith(IOService *reportingService,
    IOReportCategories categories,
    IOReportUnit unit)
{
	// fully specify the channel type for the superclass
	IOReportChannelType channelType = {
		.categories = categories,
		.report_format = kIOReportFormatSimple,
		.nelements = 1,
		.element_idx = 0
	};

	return super::init(reportingService, channelType, unit);
}


IOReturn
IOSimpleReporter::setValue(uint64_t channel_id,
    int64_t value)
{
	IOReturn res = kIOReturnError;
	IOSimpleReportValues simple_values;
	int element_index = 0;

	lockReporter();

	if (getFirstElementIndex(channel_id, &element_index) != kIOReturnSuccess) {
		res = kIOReturnBadArgument;
		goto finish;
	}


	if (copyElementValues(element_index, (IOReportElementValues *)&simple_values) != kIOReturnSuccess) {
		res = kIOReturnBadArgument;
		goto finish;
	}

	simple_values.simple_value = value;
	res = setElementValues(element_index, (IOReportElementValues *)&simple_values);

finish:
	unlockReporter();
	return res;
}


IOReturn
IOSimpleReporter::incrementValue(uint64_t channel_id,
    int64_t increment)
{
	IOReturn res = kIOReturnError;
	IOSimpleReportValues simple_values;
	int element_index = 0;

	lockReporter();

	if (getFirstElementIndex(channel_id, &element_index) != kIOReturnSuccess) {
		res = kIOReturnBadArgument;
		goto finish;
	}

	if (copyElementValues(element_index, (IOReportElementValues *)&simple_values) != kIOReturnSuccess) {
		res = kIOReturnBadArgument;
		goto finish;
	}

	simple_values.simple_value += increment;

	res = setElementValues(element_index, (IOReportElementValues *)&simple_values);

finish:
	unlockReporter();
	return res;
}

int64_t
IOSimpleReporter::getValue(uint64_t channel_id)
{
	IOSimpleReportValues *values = NULL;
	int64_t simple_value = (int64_t)kIOReportInvalidValue;
	int index = 0;

	lockReporter();

	if (getFirstElementIndex(channel_id, &index) == kIOReturnSuccess) {
		values = (IOSimpleReportValues *)getElementValues(index);

		if (values != NULL) {
			simple_value = values->simple_value;
		}
	}

	unlockReporter();
	return simple_value;
}

/* static */ OSPtr<IOReportLegendEntry>
IOSimpleReporter::createLegend(const uint64_t *channelIDs,
    const char **channelNames,
    int channelCount,
    IOReportCategories categories,
    IOReportUnit unit)
{
	IOReportChannelType channelType = {
		.categories = categories,
		.report_format = kIOReportFormatSimple,
		.nelements = 1,
		.element_idx = 0
	};

	return IOReporter::legendWith(channelIDs, channelNames, channelCount, channelType, unit);
}
