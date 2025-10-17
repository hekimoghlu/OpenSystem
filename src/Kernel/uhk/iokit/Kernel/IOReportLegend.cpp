/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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

#include <IOKit/IOKernelReportStructs.h>
#include <IOKit/IOKernelReporters.h>


//#define IORDEBUG_LEGEND 1

#ifdef IORDEBUG_LEGEND
    #define IORLEGENDLOG(fmt, args...)      \
    do {                                    \
	IOLog("IOReportLegend | ");         \
	IOLog(fmt, ##args);                 \
	IOLog("\n");                        \
    } while(0)
#else
    #define IORLEGENDLOG(fmt, args...)
#endif


#define super OSObject
OSDefineMetaClassAndStructors(IOReportLegend, OSObject);

OSSharedPtr<IOReportLegend>
IOReportLegend::with(OSArray *legend)
{
	OSSharedPtr<IOReportLegend> iorLegend = OSMakeShared<IOReportLegend>();

	if (iorLegend) {
		if (legend != NULL) {
			if (iorLegend->initWith(legend) != kIOReturnSuccess) {
				return nullptr;
			}
		}

		return iorLegend;
	} else {
		return nullptr;
	}
}

/* must clean up everything if it fails */
IOReturn
IOReportLegend::initWith(OSArray *legend)
{
	if (legend) {
		_reportLegend = OSArray::withArray(legend);
	}

	if (_reportLegend == NULL) {
		return kIOReturnError;
	} else {
		return kIOReturnSuccess;
	}
}


void
IOReportLegend::free(void)
{
	super::free();
}


OSArray*
IOReportLegend::getLegend(void)
{
	return _reportLegend.get();
}

IOReturn
IOReportLegend::addReporterLegend(IOService *reportingService,
    IOReporter *reporter,
    const char *groupName,
    const char *subGroupName)
{
	IOReturn res = kIOReturnError;
	OSSharedPtr<IOReportLegend> legend;
	OSSharedPtr<OSObject> curLegend;

	// No need to check groupName and subGroupName because optional params
	if (!reportingService || !reporter) {
		goto finish;
	}

	// It's fine if the legend doesn't exist (IOReportLegend::with(NULL)
	// is how you make an empty legend).  If it's not an array, then
	// we're just going to replace it.
	curLegend = reportingService->copyProperty(kIOReportLegendKey);
	legend = IOReportLegend::with(OSDynamicCast(OSArray, curLegend.get()));
	if (!legend) {
		goto finish;
	}

	// Add the reporter's entries and update the service property.
	// The overwrite triggers a release of the old legend array.
	legend->addReporterLegend(reporter, groupName, subGroupName);
	reportingService->setProperty(kIOReportLegendKey, legend->getLegend());
	reportingService->setProperty(kIOReportLegendPublicKey, true);

	res = kIOReturnSuccess;

finish:
	return res;
}


IOReturn
IOReportLegend::addLegendEntry(IOReportLegendEntry *legendEntry,
    const char *groupName,
    const char *subGroupName)
{
	kern_return_t res = kIOReturnError;
	OSSharedPtr<const OSSymbol> tmpGroupName;
	OSSharedPtr<const OSSymbol> tmpSubGroupName;

	if (!legendEntry) {
		return res;
	}

	if (groupName) {
		tmpGroupName = OSSymbol::withCString(groupName);
	}

	if (subGroupName) {
		tmpSubGroupName = OSSymbol::withCString(subGroupName);
	}

	// It is ok to call appendLegendWith() if tmpGroups are NULL
	res = organizeLegend(legendEntry, tmpGroupName.get(), tmpSubGroupName.get());

	return res;
}


IOReturn
IOReportLegend::addReporterLegend(IOReporter *reporter,
    const char *groupName,
    const char *subGroupName)
{
	IOReturn res = kIOReturnError;
	OSSharedPtr<IOReportLegendEntry> legendEntry;

	if (reporter) {
		legendEntry = reporter->createLegend();

		if (legendEntry) {
			res = addLegendEntry(legendEntry.get(), groupName, subGroupName);
		}
	}

	return res;
}


IOReturn
IOReportLegend::organizeLegend(IOReportLegendEntry *legendEntry,
    const OSSymbol *groupName,
    const OSSymbol *subGroupName)
{
	if (!legendEntry) {
		return kIOReturnBadArgument;
	}

	if (!groupName && subGroupName) {
		return kIOReturnBadArgument;
	}

	IORLEGENDLOG("IOReportLegend::organizeLegend");
	// Legend is empty, enter first node
	if (_reportLegend == NULL) {
		IORLEGENDLOG("IOReportLegend::new legend creation");
		_reportLegend = OSArray::withCapacity(1);

		if (!_reportLegend) {
			return kIOReturnNoMemory;
		}
	}

	if (groupName) {
		legendEntry->setObject(kIOReportLegendGroupNameKey, groupName);
	}

	if (subGroupName) {
		legendEntry->setObject(kIOReportLegendSubGroupNameKey, subGroupName);
	}

	_reportLegend->setObject(legendEntry);

	// callers can now safely release legendEntry (it is part of _reportLegend)

	return kIOReturnSuccess;
}
