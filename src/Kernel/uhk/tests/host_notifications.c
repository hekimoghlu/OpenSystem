/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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
#include <sys/time.h>
#include <mach/mach.h>
#include <mach/mach_host.h>

#include <darwintest.h>

T_GLOBAL_META(
	T_META_CHECK_LEAKS(false),
	T_META_LTEPHASE(LTE_POSTINIT));

static void
do_test(int notify_type, void (^trigger_block)(void))
{
	mach_port_t port;
	T_ASSERT_MACH_SUCCESS(mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &port), NULL);

	T_ASSERT_MACH_SUCCESS(host_request_notification(mach_host_self(), notify_type, port), NULL);

	trigger_block();

	struct {
		mach_msg_header_t hdr;
		mach_msg_trailer_t trailer;
	} message = { .hdr = {
			      .msgh_bits = 0,
			      .msgh_size = sizeof(mach_msg_header_t),
			      .msgh_remote_port = MACH_PORT_NULL,
			      .msgh_local_port = port,
			      .msgh_voucher_port = MACH_PORT_NULL,
			      .msgh_id = 0,
		      }};

	T_ASSERT_EQ(MACH_RCV_TOO_LARGE, mach_msg_receive(&message.hdr), NULL);
	mach_msg_destroy(&message.hdr);
}

T_DECL(host_notify_calendar_change, "host_request_notification(HOST_NOTIFY_CALENDAR_CHANGE)")
{
	do_test(HOST_NOTIFY_CALENDAR_CHANGE, ^{
		struct timeval tm;
		if (gettimeofday(&tm, NULL) != 0 || settimeofday(&tm, NULL) != 0) {
		        T_SKIP("Unable to settimeofday()");
		}
	});
}

T_DECL(host_notify_calendar_set, "host_request_notification(HOST_NOTIFY_CALENDAR_SET)")
{
	do_test(HOST_NOTIFY_CALENDAR_SET, ^{
		struct timeval tm;
		if (gettimeofday(&tm, NULL) != 0 || settimeofday(&tm, NULL) != 0) {
		        T_SKIP("Unable to settimeofday()");
		}
	});
}


T_DECL(host_notify_twice, "host_request_notification(HOST_NOTIFY_CALENDAR_SET)")
{
	mach_port_t port;

	T_ASSERT_MACH_SUCCESS(mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &port), NULL);

	T_ASSERT_MACH_SUCCESS(host_request_notification(mach_host_self(), HOST_NOTIFY_CALENDAR_SET, port),
	    "first registration succeeds");
	T_ASSERT_MACH_ERROR(host_request_notification(mach_host_self(), HOST_NOTIFY_CALENDAR_CHANGE, port),
	    KERN_INVALID_CAPABILITY, "second registration fails");
}
