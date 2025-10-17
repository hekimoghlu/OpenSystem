/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#include <IOKit/IOKitLib.h>
#include <Kernel/IOKit/crypto/AppleKeyStoreDefs.h>


T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

T_DECL(ioconnectasyncmethod_referenceCnt,
    "Test IOConnectCallAsyncMethod with referenceCnt < 1",
    T_META_ASROOT(true))
{
	io_service_t service;
	io_connect_t conn;
	mach_port_t wakePort;
	uint64_t reference = 0;
	service = IOServiceGetMatchingService(kIOMasterPortDefault, IOServiceMatching(kAppleKeyStoreServiceName));
	if (service == IO_OBJECT_NULL) {
		T_SKIP("Service " kAppleKeyStoreServiceName " could not be opened. skipping test");
	}
	T_ASSERT_NE(service, MACH_PORT_NULL, "got " kAppleKeyStoreServiceName " service");
	T_ASSERT_MACH_SUCCESS(IOServiceOpen(service, mach_task_self(), 0, &conn), "opened connection to service");
	T_ASSERT_MACH_SUCCESS(mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &wakePort), "allocated wake port");
	T_ASSERT_MACH_ERROR(IOConnectCallAsyncMethod(conn, 0 /* selector */, wakePort, &reference, 0 /* referenceCnt */,
	    NULL /* input */, 0 /* inputCnt */, NULL /* inputStruct */, 0 /* inputStructCnt */,
	    NULL /* output */, 0 /* outputCnt */, NULL /* outputStruct */, 0 /* outputStructCntP */), kIOReturnBadArgument, "IOConnectCallAsyncMethod should fail with kIOReturnBadArgument");
	IOServiceClose(conn);
	mach_port_mod_refs(mach_task_self(), wakePort, MACH_PORT_RIGHT_RECEIVE, -1);
}
