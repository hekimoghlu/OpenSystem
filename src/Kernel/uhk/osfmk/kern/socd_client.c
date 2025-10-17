/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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
#include <kern/clock.h>
#include <kern/cpu_data.h>
#include <kern/debug.h>
#include <kern/socd_client.h>
#include <kern/startup.h>
#include <os/overflow.h>
#include <os/atomic_private.h>
#include <libkern/section_keywords.h>

// #define SOCD_CLIENT_HDR_VERSION 0x1 // original implementation
#define SOCD_CLIENT_HDR_VERSION 0x2 // add 'mode' bits to debugid to support sticky tracepoints

/* Configuration values mutable only at init time */
typedef struct {
	uint64_t boot_time_ns;
	vm_offset_t trace_buff_offset;
	uint32_t trace_buff_len;
} socd_client_cfg_t;

static SECURITY_READ_ONLY_LATE(socd_client_cfg_t) socd_client_cfg = {0};
static SECURITY_READ_ONLY_LATE(bool) socd_client_trace_available = false;
static SECURITY_READ_WRITE(bool) socd_client_trace_has_sticky_events = false;

/* Run-time state */
static struct {
	_Atomic uint32_t trace_idx;
} socd_client_state = {0};

static void
socd_client_init(void)
{
	socd_client_hdr_t hdr = {0};
	bool already_initialized = os_atomic_load(&socd_client_trace_available, relaxed);

	if (!already_initialized) {
		vm_size_t buff_size;
		vm_size_t trace_buff_size;

		buff_size = PE_init_socd_client();
		if (!buff_size) {
			return;
		}

		if (os_sub_overflow(buff_size, sizeof(hdr), &trace_buff_size)) {
			panic("socd buffer size is too small");
		}

		absolutetime_to_nanoseconds(mach_continuous_time(), &(socd_client_cfg.boot_time_ns));
		socd_client_cfg.trace_buff_offset = sizeof(hdr);
		socd_client_cfg.trace_buff_len = (uint32_t)(trace_buff_size / sizeof(socd_client_trace_entry_t));
	}

	hdr.version = SOCD_CLIENT_HDR_VERSION;
	hdr.boot_time = socd_client_cfg.boot_time_ns;
	memcpy(&hdr.kernel_uuid, kernel_uuid, sizeof(hdr.kernel_uuid));
	PE_write_socd_client_buffer(0, &hdr, sizeof(hdr));
	if (!already_initialized) {
		os_atomic_store(&socd_client_trace_available, true, release);
	}
}
STARTUP(PMAP_STEAL, STARTUP_RANK_FIRST, socd_client_init);

static void
socd_client_set_primary_kernelcache_uuid(void)
{
	long available = os_atomic_load(&socd_client_trace_available, relaxed);
	if (kernelcache_uuid_valid && available) {
		PE_write_socd_client_buffer(offsetof(socd_client_hdr_t, primary_kernelcache_uuid), &kernelcache_uuid, sizeof(kernelcache_uuid));
	}
}
STARTUP(EARLY_BOOT, STARTUP_RANK_FIRST, socd_client_set_primary_kernelcache_uuid);

void
socd_client_reinit(void)
{
	socd_client_init();
	socd_client_set_primary_kernelcache_uuid();
}

void
socd_client_trace(
	uint32_t                 debugid,
	socd_client_trace_arg_t  arg1,
	socd_client_trace_arg_t  arg2,
	socd_client_trace_arg_t  arg3,
	socd_client_trace_arg_t  arg4)
{
	socd_client_trace_entry_t entry;
	uint32_t trace_idx, buff_idx, len;
	uint64_t time_ns;
	long available;
	vm_offset_t offset;
	bool has_sticky;
	uint32_t tries = 0;

	available = os_atomic_load(&socd_client_trace_available, dependency);

	if (__improbable(!available)) {
		return;
	}

	len = os_atomic_load_with_dependency_on(&socd_client_cfg.trace_buff_len, available);
	offset = os_atomic_load_with_dependency_on(&socd_client_cfg.trace_buff_offset, available);
	has_sticky = os_atomic_load_with_dependency_on(&socd_client_trace_has_sticky_events, available);

	/* protect against the case where the buffer is full of sticky events */
	while (tries++ < len) {
		/* trace_idx is allowed to overflow */
		trace_idx = os_atomic_inc_orig(&socd_client_state.trace_idx, relaxed);
		buff_idx = trace_idx % len;

		/* if there are no sticky events then we don't need the read */
		if (has_sticky) {
			/* skip if this slot is sticky.  Read only the debugid to reduce perf impact */
			PE_read_socd_client_buffer(offset + (buff_idx * sizeof(entry)) + offsetof(socd_client_trace_entry_t, debugid), &(entry.debugid), sizeof(entry.debugid));
			if (SOCD_TRACE_EXTRACT_MODE(entry.debugid) & SOCD_TRACE_MODE_STICKY_TRACEPOINT) {
				continue;
			}
		}

		/* slot is available, write it */
		absolutetime_to_nanoseconds(mach_continuous_time(), &time_ns);
		entry.timestamp = time_ns;
		entry.debugid = debugid;
		entry.arg1 = arg1;
		entry.arg2 = arg2;
		entry.arg3 = arg3;
		entry.arg4 = arg4;

		PE_write_socd_client_buffer(offset + (buff_idx * sizeof(entry)), &entry, sizeof(entry));

		if (SOCD_TRACE_EXTRACT_MODE(entry.debugid) & SOCD_TRACE_MODE_STICKY_TRACEPOINT) {
			os_atomic_store(&socd_client_trace_has_sticky_events, true, relaxed);
		}

		break;
	}

	/* Duplicate tracepoint to kdebug */
	if (!debug_is_current_cpu_in_panic_state()) {
		KDBG(debugid, arg1, arg2, arg3, arg4);
	}
}
