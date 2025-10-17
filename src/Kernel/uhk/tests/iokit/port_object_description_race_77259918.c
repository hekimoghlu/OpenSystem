/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 26, 2024.
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
#include <unistd.h>

#include <IOKit/IOKitLib.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.iokit"),
	T_META_RUN_CONCURRENTLY(true),
	T_META_ASROOT(true),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("IOKit"),
	T_META_OWNER("souvik_b"));

struct notification_create_thread_args {
	size_t iterations;
};

static void *
notification_create_thread(void * data)
{
	struct notification_create_thread_args * args = (struct notification_create_thread_args *)data;

	for (size_t x = 0; x < args->iterations; x++) {
		IONotificationPortRef notifyPort = IONotificationPortCreate(kIOMainPortDefault);
		io_iterator_t notification;
		IOServiceAddMatchingNotification(notifyPort, kIOTerminatedNotification, IOServiceMatching("IOResources"), NULL, NULL, &notification);
		usleep(5000);
		IOObjectRelease(notification);
		IONotificationPortDestroy(notifyPort);
	}

	return NULL;
}

struct iokit_port_object_description_thread_args {
	size_t iterations;
};

static void *
iokit_port_object_description_thread(void * data)
{
	struct iokit_port_object_description_thread_args * args = (struct iokit_port_object_description_thread_args *)data;

	for (size_t x = 0; x < args->iterations; x++) {
		mach_port_name_array_t portNameArray;
		mach_msg_type_number_t portNameCount;
		mach_port_type_array_t portTypeArray;
		mach_msg_type_number_t portTypeCount;

		unsigned int kotype = 0;
		mach_vm_address_t kaddr;
		kobject_description_t desc;
		desc[0] = '\0';

		T_QUIET; T_ASSERT_MACH_SUCCESS(mach_port_names(mach_task_self(), &portNameArray, &portNameCount, &portTypeArray, &portTypeCount), "mach_port_names");
		for (size_t i = 0; i < portNameCount; i++) {
			mach_port_name_t port = portNameArray[i];
			/* the return value doesn't matter */
			(void)mach_port_kobject_description(mach_task_self(), port, &kotype, &kaddr, desc);
		}

		T_QUIET; T_ASSERT_MACH_SUCCESS(vm_deallocate(mach_task_self(), (vm_address_t) portNameArray, portNameCount * sizeof(*portNameArray)), "vm_deallocate port name array");
		T_QUIET; T_ASSERT_MACH_SUCCESS(vm_deallocate(mach_task_self(), (vm_address_t) portTypeArray, portTypeCount * sizeof(*portTypeArray)), "vm_deallocate port type array");
	}

	return NULL;
}


T_DECL(port_object_description_race, "Test iokit_port_object_description() race condition", T_META_TAG_VM_PREFERRED)
{
	pthread_t notification_create_pth;
	pthread_t iokit_port_object_description_pth;
	struct notification_create_thread_args notification_create_args;
	struct iokit_port_object_description_thread_args iokit_port_object_description_args;

	notification_create_args.iterations = 2000;
	iokit_port_object_description_args.iterations = 2000;

	// start racing threads
	T_ASSERT_POSIX_ZERO(pthread_create(&notification_create_pth, NULL, notification_create_thread, &notification_create_args), "create notification thread");
	T_ASSERT_POSIX_ZERO(pthread_create(&iokit_port_object_description_pth, NULL, iokit_port_object_description_thread, &iokit_port_object_description_args), "create port description thread");

	// wait for threads to finish
	T_ASSERT_POSIX_ZERO(pthread_join(notification_create_pth, NULL), "join notificiation thread");
	T_ASSERT_POSIX_ZERO(pthread_join(iokit_port_object_description_pth, NULL), "join port description thread");
}
