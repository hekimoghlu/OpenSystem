/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
#include <string.h>
#include <stdlib.h>
#include <mach/mach.h>
#include <mach_debug/mach_debug.h>
#include <darwintest.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.vm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("zalloc"),
	T_META_CHECK_LEAKS(false),
	T_META_RUN_CONCURRENTLY(true)
	);

static void run_test(void);

static void
run_test(void)
{
	kern_return_t kr;
	uint64_t size, i;
	mach_zone_name_t *name = NULL;
	unsigned int nameCnt = 0;
	mach_zone_info_t *info = NULL;
	unsigned int infoCnt = 0;
	mach_memory_info_t *wiredInfo = NULL;
	unsigned int wiredInfoCnt = 0;
	const char kalloc_str[] = "kalloc.";
	const char type_str[] = "type";
	size_t kt_name_len = strlen(kalloc_str) + strlen(type_str);

	kr = mach_memory_info(mach_host_self(),
	    &name, &nameCnt, &info, &infoCnt,
	    &wiredInfo, &wiredInfoCnt);
	T_QUIET; T_ASSERT_MACH_SUCCESS(kr, "mach_memory_info");
	T_QUIET; T_ASSERT_EQ(nameCnt, infoCnt, "zone name and info counts don't match");

	/* Match the names of the kalloc zones against their element sizes. */
	for (i = 0; i < nameCnt; i++) {
		const char *z_name = &name[i].mzn_name;
		if (strncmp(z_name, kalloc_str, strlen(kalloc_str)) == 0) {
			const char *size_ptr = strrchr(z_name, '.') + 1;
			T_QUIET; T_ASSERT_NOTNULL(size_ptr, "couldn't find size in name");
			size = strtoul(size_ptr, NULL, 10);
			T_LOG("ZONE NAME: %-25s ELEMENT SIZE: %llu", z_name, size);
			T_QUIET; T_ASSERT_EQ(size, info[i].mzi_elem_size, "kalloc zone name and element size don't match");
		}
	}

	if ((name != NULL) && (nameCnt != 0)) {
		kr = vm_deallocate(mach_task_self(), (vm_address_t) name,
		    (vm_size_t) (nameCnt * sizeof *name));
		T_QUIET; T_ASSERT_MACH_SUCCESS(kr, "vm_deallocate name");
	}

	if ((info != NULL) && (infoCnt != 0)) {
		kr = vm_deallocate(mach_task_self(), (vm_address_t) info,
		    (vm_size_t) (infoCnt * sizeof *info));
		T_QUIET; T_ASSERT_MACH_SUCCESS(kr, "vm_deallocate info");
	}

	if ((wiredInfo != NULL) && (wiredInfoCnt != 0)) {
		kr = vm_deallocate(mach_task_self(), (vm_address_t) wiredInfo,
		    (vm_size_t) (wiredInfoCnt * sizeof *wiredInfo));
		T_QUIET; T_ASSERT_MACH_SUCCESS(kr, "vm_deallocate wiredInfo");
	}

	T_END;
}

T_DECL( verify_kalloc_config,
    "verifies that the kalloc zones are configured correctly",
    T_META_ASROOT(true), T_META_TAG_VM_PREFERRED)
{
	run_test();
}
