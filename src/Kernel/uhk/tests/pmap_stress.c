/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 19, 2023.
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
#include <sys/sysctl.h>
#include <assert.h>
#include "test_utils.h"

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.arm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("arm"),
	T_META_OWNER("jharmening"),
	T_META_RUN_CONCURRENTLY(true),
	XNU_T_META_SOC_SPECIFIC);

T_DECL(pmap_enter_disconnect,
    "Test that a physical page can be safely mapped concurrently with a disconnect of the same page", T_META_TAG_VM_NOT_ELIGIBLE)
{
	int num_loops = 10000;
	T_ASSERT_POSIX_SUCCESS(sysctlbyname("kern.pmap_enter_disconnect_test", NULL, NULL, &num_loops, sizeof(num_loops)),
	    "kern.pmap_enter_disconnect_test, %d loops", num_loops);
}

T_DECL(pmap_exec_remove_test,
    "Test that an executable mapping can be created while another mapping of the same physical page is removed", T_META_TAG_VM_NOT_ELIGIBLE)
{
	int num_loops = 10000;
	T_ASSERT_POSIX_SUCCESS(sysctlbyname("kern.pmap_exec_remove_test", NULL, NULL, &num_loops, sizeof(num_loops)),
	    "kern.pmap_exec_remove_test, %d loops", num_loops);
}

T_DECL(pmap_compress_remove_test,
    "Test that a page can be disconnected for compression while concurrently unmapping the same page", T_META_TAG_VM_NOT_ELIGIBLE)
{
	int num_loops = 1000000;
	T_ASSERT_POSIX_SUCCESS(sysctlbyname("kern.pmap_compress_remove_test", NULL, NULL, &num_loops, sizeof(num_loops)),
	    "kern.pmap_compress_remove_test, %d loops", num_loops);
}

T_DECL(pmap_nesting_test,
    "Test that pmap_nest() and pmap_unnest() work reliably when concurrently invoked from multiple threads", T_META_TAG_VM_NOT_ELIGIBLE)
{
	int num_loops = 5;
	T_ASSERT_POSIX_SUCCESS(sysctlbyname("kern.pmap_nesting_test", NULL, NULL, &num_loops, sizeof(num_loops)),
	    "kern.pmap_nesting_test, %d loops", num_loops);
}

T_DECL(pmap_iommu_disconnect_test,
    "Test that CPU mappings of a physical page can safely be disconnected in the presence of IOMMU mappings", T_META_TAG_VM_NOT_ELIGIBLE)
{
	int run = 1;
	T_ASSERT_POSIX_SUCCESS(sysctlbyname("kern.pmap_iommu_disconnect_test", NULL, NULL, &run, sizeof(run)),
	    "kern.pmap_iommu_disconnect_test");
}

T_DECL(pmap_extended_test,
    "Test various pmap lifecycle calls in the presence of special configurations such as 4K and stage-2", T_META_TAG_VM_NOT_ELIGIBLE)
{
	int run = 1;
	T_ASSERT_POSIX_SUCCESS(sysctlbyname("kern.pmap_extended_test", NULL, NULL, &run, sizeof(run)),
	    "kern.pmap_extended_test");
}

T_DECL(pmap_huge_pv_list_test,
    "Test that extremely large PV lists can be managed without spinlock timeouts or other panics",
    T_META_REQUIRES_SYSCTL_EQ("kern.page_protection_type", 2), T_META_TAG_VM_NOT_ELIGIBLE)
{
	struct {
		unsigned int num_loops;
		unsigned int num_mappings;
	} hugepv_in;
	hugepv_in.num_loops = 500;
	hugepv_in.num_mappings = 500000;
	T_ASSERT_POSIX_SUCCESS(sysctlbyname("kern.pmap_huge_pv_list_test", NULL, NULL,
	    &hugepv_in, sizeof(hugepv_in)), "kern.pmap_huge_pv_list_test");
}

T_DECL(pmap_reentrance_test,
    "Test that the pmap can be reentered by an async exception handler",
    T_META_REQUIRES_SYSCTL_EQ("kern.page_protection_type", 2), T_META_TAG_VM_NOT_ELIGIBLE)
{
	int num_loops = 10000;
	T_ASSERT_POSIX_SUCCESS(sysctlbyname("kern.pmap_reentrance_test", NULL, NULL, &num_loops, sizeof(num_loops)),
	    "kern.pmap_reentrance_test, %d loops", num_loops);
}
