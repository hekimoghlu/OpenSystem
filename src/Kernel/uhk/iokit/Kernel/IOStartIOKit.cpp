/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 8, 2021.
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
#include <libkern/c++/OSUnserialize.h>
#include <libkern/c++/OSKext.h>
#include <libkern/section_keywords.h>
#include <libkern/version.h>
#include <IOKit/IORegistryEntry.h>
#include <IOKit/IODeviceTreeSupport.h>
#include <IOKit/IOCatalogue.h>
#include <IOKit/IOUserClient.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <IOKit/IOPlatformExpert.h>
#include <IOKit/IOKernelReporters.h>
#include <IOKit/IOLib.h>
#include <IOKit/IOKitKeys.h>
#include <IOKit/IOKitDebug.h>
#include <IOKit/pwr_mgt/RootDomain.h>
#include <IOKit/pwr_mgt/IOPMinformeeList.h>
#include <IOKit/IOStatisticsPrivate.h>
#include <IOKit/IOKitKeysPrivate.h>
#include <IOKit/IOInterruptAccountingPrivate.h>
#include <IOKit/assert.h>
#include <sys/conf.h>

#include "IOKitKernelInternal.h"

const OSSymbol * gIOProgressBackbufferKey;
OSSet *          gIORemoveOnReadProperties;

extern "C" {
void InitIOKit(void *dtTop);
void ConfigureIOKit(void);
void StartIOKitMatching(void);
void IORegistrySetOSBuildVersion(char * build_version);
void IORecordProgressBackbuffer(void * buffer, size_t size, uint32_t theme);

extern void OSlibkernInit(void);

void iokit_post_constructor_init(void);

SECURITY_READ_ONLY_LATE(static IOPlatformExpertDevice*) gRootNub;

#include <kern/clock.h>
#include <sys/time.h>

void
IOKitInitializeTime( void )
{
	mach_timespec_t         t;

	t.tv_sec = 30;
	t.tv_nsec = 0;

	IOService::waitForService(
		IOService::resourceMatching("IORTC"), &t );
#if defined(__i386__) || defined(__x86_64__)
	IOService::waitForService(
		IOService::resourceMatching("IONVRAM"), &t );
#endif

	clock_initialize_calendar();
}

void
iokit_post_constructor_init(void)
{
	IORegistryEntry *           root;
	OSObject *                  obj;

	IOCPUInitialize();
	IOPlatformActionsInitialize();
	root = IORegistryEntry::initialize();
	assert( root );
	IOService::initialize();
	IOCatalogue::initialize();
	IOStatistics::initialize();
	OSKext::initialize();
	IOUserClient::initialize();
	IOMemoryDescriptor::initialize();
	IORootParent::initialize();
	IOReporter::initialize();

	// Initializes IOPMinformeeList class-wide shared lock
	IOPMinformeeList::getSharedRecursiveLock();

	obj = OSString::withCString( version );
	assert( obj );
	if (obj) {
		root->setProperty( kIOKitBuildVersionKey, obj );
		obj->release();
	}
	obj = IOKitDiagnostics::diagnostics();
	if (obj) {
		root->setProperty( kIOKitDiagnosticsKey, obj );
		obj->release();
	}
}

/*****
 * Pointer into bootstrap KLD segment for functions never used past startup.
 */
void (*record_startup_extensions_function)(void) = NULL;

void
InitIOKit(void *dtTop)
{
	// Compat for boot-args
	gIOKitTrace |= (gIOKitDebug & kIOTraceCompatBootArgs);

	//
	// Have to start IOKit environment before we attempt to start
	// the C++ runtime environment.  At some stage we have to clean up
	// the initialisation path so that OS C++ can initialise independantly
	// of iokit basic service initialisation, or better we have IOLib stuff
	// initialise as basic OS services.
	//
	IOLibInit();
	OSlibkernInit();
	IOMachPortInitialize();

	gIOProgressBackbufferKey  = OSSymbol::withCStringNoCopy(kIOProgressBackbufferKey);
	gIORemoveOnReadProperties = OSSet::withObjects((const OSObject **) &gIOProgressBackbufferKey, 1);

	interruptAccountingInit();

	gRootNub = new IOPlatformExpertDevice;
	if (__improbable(gRootNub == NULL)) {
		panic("Failed to allocate IOKit root nub");
	}
	bool ok = gRootNub->init(dtTop);
	if (__improbable(!ok)) {
		panic("Failed to initialize IOKit root nub");
	}
	gRootNub->attach(NULL);

	/* If the bootstrap segment set up a function to record startup
	 * extensions, call it now.
	 */
	if (record_startup_extensions_function) {
		record_startup_extensions_function();
	}
}

void
ConfigureIOKit(void)
{
	assert(gRootNub != NULL);
	gRootNub->configureDefaults();
}

void
StartIOKitMatching(void)
{
	SOCD_TRACE_XNU(START_IOKIT, SOCD_TRACE_MODE_NONE);
	assert(gRootNub != NULL);
	bool ok = gRootNub->startIOServiceMatching();
	if (__improbable(!ok)) {
		panic("Failed to start IOService matching");
	}

#if !NO_KEXTD
	if (OSKext::iokitDaemonAvailable()) {
		/* Add a busy count to keep the registry busy until the IOKit daemon has
		 * completely finished launching. This is decremented when the IOKit daemon
		 * messages the kernel after the in-kernel linker has been
		 * removed and personalities have been sent.
		 */
		IOService::getServiceRoot()->adjustBusy(1);
	}
#endif
}

void
IORegistrySetOSBuildVersion(char * build_version)
{
	IORegistryEntry * root = IORegistryEntry::getRegistryRoot();

	if (root) {
		if (build_version) {
			root->setProperty(kOSBuildVersionKey, build_version);
		} else {
			root->removeProperty(kOSBuildVersionKey);
		}
	}

	return;
}

void
IORecordProgressBackbuffer(void * buffer, size_t size, uint32_t theme)
{
	IORegistryEntry * chosen;

	if (((unsigned int) size) != size) {
		return;
	}
	if ((chosen = IORegistryEntry::fromPath(kIODeviceTreePlane ":/chosen"))) {
		chosen->setProperty(kIOProgressBackbufferKey, buffer, (unsigned int) size);
		chosen->setProperty(kIOProgressColorThemeKey, theme, 32);

		chosen->release();
	}
}
}; /* extern "C" */
