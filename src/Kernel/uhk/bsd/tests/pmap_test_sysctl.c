/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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
#include <sys/sysctl.h>

extern kern_return_t test_pmap_enter_disconnect(unsigned int);
extern kern_return_t test_pmap_compress_remove(unsigned int);
extern kern_return_t test_pmap_exec_remove(unsigned int);
extern kern_return_t test_pmap_nesting(unsigned int);
extern kern_return_t test_pmap_iommu_disconnect(void);
extern kern_return_t test_pmap_extended(void);
extern void test_pmap_call_overhead(unsigned int);
extern uint64_t test_pmap_page_protect_overhead(unsigned int, unsigned int);
#if CONFIG_SPTM
extern kern_return_t test_pmap_huge_pv_list(unsigned int, unsigned int);
extern kern_return_t test_pmap_reentrance(unsigned int);
#endif

static int
sysctl_test_pmap_enter_disconnect(__unused struct sysctl_oid *oidp, __unused void *arg1, __unused int arg2, struct sysctl_req *req)
{
	unsigned int num_loops;
	int error, changed;
	error = sysctl_io_number(req, 0, sizeof(num_loops), &num_loops, &changed);
	if (error || !changed) {
		return error;
	}
	return test_pmap_enter_disconnect(num_loops);
}

SYSCTL_PROC(_kern, OID_AUTO, pmap_enter_disconnect_test,
    CTLTYPE_INT | CTLFLAG_RW | CTLFLAG_LOCKED,
    0, 0, sysctl_test_pmap_enter_disconnect, "I", "");

static int
sysctl_test_pmap_compress_remove(__unused struct sysctl_oid *oidp, __unused void *arg1, __unused int arg2, struct sysctl_req *req)
{
	unsigned int num_loops;
	int error, changed;
	error = sysctl_io_number(req, 0, sizeof(num_loops), &num_loops, &changed);
	if (error || !changed) {
		return error;
	}
	return test_pmap_compress_remove(num_loops);
}

SYSCTL_PROC(_kern, OID_AUTO, pmap_compress_remove_test,
    CTLTYPE_INT | CTLFLAG_RW | CTLFLAG_LOCKED,
    0, 0, sysctl_test_pmap_compress_remove, "I", "");

static int
sysctl_test_pmap_exec_remove(__unused struct sysctl_oid *oidp, __unused void *arg1, __unused int arg2, struct sysctl_req *req)
{
	unsigned int num_loops;
	int error, changed;
	error = sysctl_io_number(req, 0, sizeof(num_loops), &num_loops, &changed);
	if (error || !changed) {
		return error;
	}
	return test_pmap_exec_remove(num_loops);
}

SYSCTL_PROC(_kern, OID_AUTO, pmap_exec_remove_test,
    CTLTYPE_INT | CTLFLAG_RW | CTLFLAG_LOCKED,
    0, 0, sysctl_test_pmap_exec_remove, "I", "");

static int
sysctl_test_pmap_nesting(__unused struct sysctl_oid *oidp, __unused void *arg1, __unused int arg2, struct sysctl_req *req)
{
	unsigned int num_loops;
	int error, changed;
	error = sysctl_io_number(req, 0, sizeof(num_loops), &num_loops, &changed);
	if (error || !changed) {
		return error;
	}
	return test_pmap_nesting(num_loops);
}
SYSCTL_PROC(_kern, OID_AUTO, pmap_nesting_test,
    CTLTYPE_INT | CTLFLAG_RW | CTLFLAG_LOCKED,
    0, 0, sysctl_test_pmap_nesting, "I", "");

static int
sysctl_test_pmap_iommu_disconnect(__unused struct sysctl_oid *oidp, __unused void *arg1, __unused int arg2, struct sysctl_req *req)
{
	unsigned int run = 0;
	int error, changed;
	error = sysctl_io_number(req, 0, sizeof(run), &run, &changed);
	if (error || !changed) {
		return error;
	}
	return test_pmap_iommu_disconnect();
}

SYSCTL_PROC(_kern, OID_AUTO, pmap_iommu_disconnect_test,
    CTLTYPE_INT | CTLFLAG_RW | CTLFLAG_LOCKED,
    0, 0, sysctl_test_pmap_iommu_disconnect, "I", "");

static int
sysctl_test_pmap_extended(__unused struct sysctl_oid *oidp, __unused void *arg1, __unused int arg2, struct sysctl_req *req)
{
	unsigned int run = 0;
	int error, changed;
	error = sysctl_io_number(req, 0, sizeof(run), &run, &changed);
	if (error || !changed) {
		return error;
	}
	return test_pmap_extended();
}

SYSCTL_PROC(_kern, OID_AUTO, pmap_extended_test,
    CTLTYPE_INT | CTLFLAG_RW | CTLFLAG_LOCKED,
    0, 0, sysctl_test_pmap_extended, "I", "");

static int
sysctl_test_pmap_call_overhead(__unused struct sysctl_oid *oidp, __unused void *arg1, __unused int arg2, struct sysctl_req *req)
{
	unsigned int num_loops;
	int error, changed;
	error = sysctl_io_number(req, 0, sizeof(num_loops), &num_loops, &changed);
	if (error || !changed) {
		return error;
	}
	test_pmap_call_overhead(num_loops);
	return 0;
}

SYSCTL_PROC(_kern, OID_AUTO, pmap_call_overhead_test,
    CTLTYPE_INT | CTLFLAG_RW | CTLFLAG_LOCKED,
    0, 0, sysctl_test_pmap_call_overhead, "I", "");

static int
sysctl_test_pmap_page_protect_overhead(__unused struct sysctl_oid *oidp, __unused void *arg1, __unused int arg2, struct sysctl_req *req)
{
	struct {
		unsigned int num_loops;
		unsigned int num_aliases;
	} ppo_in;

	int error;
	uint64_t duration;

	error = SYSCTL_IN(req, &ppo_in, sizeof(ppo_in));
	if (error) {
		return error;
	}

	duration = test_pmap_page_protect_overhead(ppo_in.num_loops, ppo_in.num_aliases);
	error = SYSCTL_OUT(req, &duration, sizeof(duration));
	return error;
}

SYSCTL_PROC(_kern, OID_AUTO, pmap_page_protect_overhead_test,
    CTLTYPE_OPAQUE | CTLFLAG_RW | CTLFLAG_LOCKED, 0, 0, sysctl_test_pmap_page_protect_overhead, "-", "");

#if CONFIG_SPTM

static int
sysctl_test_pmap_huge_pv_list(__unused struct sysctl_oid *oidp, __unused void *arg1, __unused int arg2, struct sysctl_req *req)
{
	struct {
		unsigned int num_loops;
		unsigned int num_mappings;
	} hugepv_in;

	int error = SYSCTL_IN(req, &hugepv_in, sizeof(hugepv_in));
	if (error) {
		return error;
	}
	return test_pmap_huge_pv_list(hugepv_in.num_loops, hugepv_in.num_mappings);
}

SYSCTL_PROC(_kern, OID_AUTO, pmap_huge_pv_list_test,
    CTLTYPE_OPAQUE | CTLFLAG_RW | CTLFLAG_LOCKED, 0, 0, sysctl_test_pmap_huge_pv_list, "-", "");

static int
sysctl_test_pmap_reentrance(__unused struct sysctl_oid *oidp, __unused void *arg1, __unused int arg2, struct sysctl_req *req)
{
	unsigned int num_loops;
	int error, changed;
	error = sysctl_io_number(req, 0, sizeof(num_loops), &num_loops, &changed);
	if (error || !changed) {
		return error;
	}
	return test_pmap_reentrance(num_loops);
}

SYSCTL_PROC(_kern, OID_AUTO, pmap_reentrance_test,
    CTLTYPE_INT | CTLFLAG_RW | CTLFLAG_LOCKED,
    0, 0, sysctl_test_pmap_reentrance, "I", "");

#endif
