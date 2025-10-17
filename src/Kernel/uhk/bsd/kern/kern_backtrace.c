/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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
#include <kern/backtrace.h>
#include <kern/kalloc.h>
#include <sys/errno.h>
#include <sys/sysctl.h>
#include <sys/systm.h>

#if DEVELOPMENT || DEBUG

/*
 * Ignore -Wxnu-typed-allocators for this file, as it implements
 * sysctls that are only available for DEVELOPMENT || DEBUG builds.
 */
__typed_allocators_ignore_push

#define MAX_BACKTRACE  (128)

#define BACKTRACE_USER (0)
#define BACKTRACE_USER_RESUME (1)
static int backtrace_user_sysctl SYSCTL_HANDLER_ARGS;

#define BACKTRACE_KERN_TEST_PACK_UNPACK (0)
#define BACKTRACE_KERN_TEST_PACKED (1)
static int backtrace_kernel_sysctl SYSCTL_HANDLER_ARGS;

SYSCTL_NODE(_kern, OID_AUTO, backtrace, CTLFLAG_RW | CTLFLAG_LOCKED, 0,
    "backtrace");

SYSCTL_PROC(_kern_backtrace, OID_AUTO, user,
    CTLFLAG_RW | CTLFLAG_LOCKED, (void *)BACKTRACE_USER,
    sizeof(uint64_t), backtrace_user_sysctl, "O",
    "take user backtrace of current thread");

SYSCTL_PROC(_kern_backtrace, OID_AUTO, kernel_tests,
    CTLFLAG_RW | CTLFLAG_LOCKED, (void *)BACKTRACE_USER,
    sizeof(uint64_t), backtrace_kernel_sysctl, "O",
    "take user backtrace of current thread");

static int
backtrace_kernel_sysctl SYSCTL_HANDLER_ARGS
{
	unsigned int scenario = (unsigned int)req->newlen;
	uintptr_t *bt = NULL;
	uint8_t *packed_bt = NULL;
	uintptr_t *unpacked_bt = NULL;
	unsigned int bt_len = 0;
	size_t bt_size = 0;
	errno_t error = 0;

	bt_len = 24;
	bt_size = sizeof(bt[0]) * bt_len;
	bt = kalloc_data(bt_size, Z_WAITOK | Z_ZERO);
	packed_bt = kalloc_data(bt_size, Z_WAITOK | Z_ZERO);
	unpacked_bt = kalloc_data(bt_size, Z_WAITOK | Z_ZERO);
	if (!bt || !packed_bt || !unpacked_bt) {
		error = ENOBUFS;
		goto out;
	}
	backtrace_info_t info = BTI_NONE;
	unsigned int len = backtrace(bt, bt_len, NULL, &info);
	backtrace_info_t packed_info = BTI_NONE;
	size_t packed_size = 0;
	if (scenario == BACKTRACE_KERN_TEST_PACK_UNPACK) {
		packed_size = backtrace_pack(BTP_KERN_OFFSET_32, packed_bt, bt_size,
		    bt, len);
	} else {
		packed_size = backtrace_packed(BTP_KERN_OFFSET_32, packed_bt, bt_size,
		    NULL, &packed_info);
	}
	unsigned int unpacked_len = backtrace_unpack(BTP_KERN_OFFSET_32,
	    unpacked_bt, bt_len, packed_bt, packed_size);
	if (unpacked_len != len) {
		printf("backtrace_tests: length %u != %u unpacked\n", len,
		    unpacked_len);
		error = ERANGE;
		goto out;
	}
	for (unsigned int i = 0; i < len; i++) {
		if (unpacked_bt[i] != bt[i]) {
			printf("backtrace_tests: bad address %u: 0x%lx != 0x%lx unpacked",
			    i, bt[i], unpacked_bt[i]);
			error = EINVAL;
		}
	}

out:
	if (bt) {
		kfree_data(bt, bt_size);
	}
	if (packed_bt) {
		kfree_data(packed_bt, bt_size);
	}
	if (unpacked_bt) {
		kfree_data(unpacked_bt, bt_size);
	}
	return error;
}

static int
backtrace_user_sysctl SYSCTL_HANDLER_ARGS
{
#pragma unused(oidp, arg1, arg2)
	unsigned int scenario = (unsigned int)req->newlen;
	uintptr_t *bt = NULL;
	unsigned int bt_len = 0, bt_filled = 0, bt_space = 0;
	size_t bt_size = 0;
	errno_t error = 0;

	bool user_scenario = scenario == BACKTRACE_USER;
	bool resume_scenario = scenario == BACKTRACE_USER_RESUME;
	if (!user_scenario && !resume_scenario) {
		return ENOTSUP;
	}

	if (req->oldptr == USER_ADDR_NULL || req->oldlen == 0) {
		return EFAULT;
	}

	bt_len = req->oldlen > MAX_BACKTRACE ? MAX_BACKTRACE :
	    (unsigned int)req->oldlen;
	bt_size = sizeof(bt[0]) * bt_len;
	bt = kalloc_data(bt_size, Z_WAITOK | Z_ZERO);
	if (!bt) {
		return ENOBUFS;
	}
	bt_space = resume_scenario ? bt_len / 2 : bt_len;
	struct backtrace_user_info btinfo = BTUINFO_INIT;
	bt_filled = backtrace_user(bt, bt_space, NULL, &btinfo);
	error = btinfo.btui_error;
	if (error != 0) {
		goto out;
	}
	if (resume_scenario) {
		if (!(btinfo.btui_info & BTI_TRUNCATED)) {
			error = ENOSPC;
			goto out;
		}
		struct backtrace_control ctl = {
			.btc_frame_addr = btinfo.btui_next_frame_addr,
		};
		btinfo = BTUINFO_INIT;
		unsigned int bt_more = backtrace_user(bt + bt_filled, bt_space, &ctl,
		    &btinfo);
		error = btinfo.btui_error;
		if (error != 0) {
			goto out;
		}
		bt_filled += bt_more;
	}
	bt_filled = min(bt_filled, bt_len);
	if (btinfo.btui_async_frame_addr != 0 &&
	    btinfo.btui_async_start_index != 0) {
		// Put the async call stack inline after the real call stack.
		unsigned int start_index = btinfo.btui_async_start_index;
		uintptr_t frame_addr = btinfo.btui_async_frame_addr;
		unsigned int bt_left = bt_len - start_index;
		struct backtrace_control ctl = { .btc_frame_addr = frame_addr, };
		btinfo = BTUINFO_INIT;
		unsigned int async_filled = backtrace_user(bt + start_index, bt_left,
		    &ctl, &btinfo);
		error = btinfo.btui_error;
		if (error != 0) {
			goto out;
		}
		bt_filled = min(start_index + async_filled, bt_len);
	}

	error = copyout(bt, req->oldptr, sizeof(bt[0]) * bt_filled);
	if (error) {
		goto out;
	}
	req->oldidx = bt_filled;

out:
	kfree_data(bt, bt_size);
	return error;
}

__typed_allocators_ignore_pop

#endif /* DEVELOPMENT || DEBUG */
