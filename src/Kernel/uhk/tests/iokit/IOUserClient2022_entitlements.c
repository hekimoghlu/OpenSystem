/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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
#include <darwintest.h>
#include <mach/mach.h>
#include <mach/message.h>
#include <stdlib.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <signal.h>
#include <mach/mach_vm.h>

#include <IOKit/IOKitLib.h>
#include "service_helpers.h"

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.iokit"),
	T_META_RUN_CONCURRENTLY(true),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("IOKit"),
	T_META_OWNER("ayao"));

//A client like IOUserClient2022_entitlements_unentitled without the com.apple.iokit.test-check-entitlement-open entitlement should fail on IOServiceOpen
//A client like IOUserClient2022_entitlements without com.apple.iokit.test-check-entitlement-per-selector should fail to call selector 1
T_DECL(TESTNAME, "Test IOUserClient2022 entitlement enforcement")
{
	io_service_t service;
	io_connect_t conn;
	const char *serviceName = "TestIOUserClient2022Entitlements";

	T_QUIET; T_ASSERT_POSIX_SUCCESS(IOTestServiceFindService(serviceName, &service), "Find service");
	T_QUIET; T_ASSERT_NE(service, MACH_PORT_NULL, "got service");
#if OPEN_ENTITLED
	T_QUIET; T_ASSERT_MACH_SUCCESS(IOServiceOpen(service, mach_task_self(), 0, &conn), "open service");
	//We expect failure since we don't have the entitlement to use selector 1
	T_QUIET; T_ASSERT_NE(IOConnectCallMethod(conn, 1,
	    NULL, 0, NULL, 0, NULL, 0, NULL, NULL), kIOReturnSuccess, "call external method 2");
#else
	//not entitled to open the service, so we expect failure.
	T_QUIET; T_ASSERT_NE(IOServiceOpen(service, mach_task_self(), 0, &conn), kIOReturnSuccess, "open service");
#endif
	IOConnectRelease(conn);
	IOObjectRelease(service);
}
