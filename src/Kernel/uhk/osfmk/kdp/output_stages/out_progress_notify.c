/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 18, 2023.
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
#ifdef CONFIG_KDP_INTERACTIVE_DEBUGGING

#include <mach/mach_types.h>
#include <IOKit/IOTypes.h>
#include <IOKit/IOPlatformExpert.h>
#include <kdp/output_stages/output_stages.h>
#include <kdp/kdp_core.h>
#include <kdp/processor_core.h>
#include <vm/vm_kern_xnu.h>

#define NOTIFY_INTERVAL_NSECS (5 * NSEC_PER_SEC)

struct progress_notify_stage_data {
	uint64_t notify_interval_matus;
	uint64_t last_notify_timestamp;
};

static void
progress_notify_stage_reset(struct kdp_output_stage *stage)
{
	struct progress_notify_stage_data *data = (struct progress_notify_stage_data*) stage->kos_data;

	data->last_notify_timestamp = 0;
}

static kern_return_t
progress_notify_stage_outproc(struct kdp_output_stage *stage, unsigned int request,
    char *corename, uint64_t length, void * panic_data)
{
	kern_return_t err = KERN_SUCCESS;
	struct progress_notify_stage_data *data = (struct progress_notify_stage_data*) stage->kos_data;
	struct kdp_output_stage  *next_stage = STAILQ_NEXT(stage, kos_next);
	uint64_t now = mach_absolute_time();

	assert(next_stage != NULL);

	if (now >= (data->last_notify_timestamp + data->notify_interval_matus)) {
		PEHaltRestart(kPEPanicDiagnosticsInProgress);
		data->last_notify_timestamp = now;
	}

	err = next_stage->kos_funcs.kosf_outproc(next_stage, request, corename, length, panic_data);
	if (KERN_SUCCESS != err) {
		kern_coredump_log(NULL, "%s (during forwarding) returned 0x%x\n", __func__, err);
		return err;
	}

	return KERN_SUCCESS;
}

static void
progress_notify_stage_free(struct kdp_output_stage *stage)
{
	kmem_free(kernel_map, (vm_offset_t) stage->kos_data, stage->kos_data_size);

	stage->kos_data = NULL;
	stage->kos_data_size = 0;
	stage->kos_initialized = false;
}

kern_return_t
progress_notify_stage_initialize(struct kdp_output_stage *stage)
{
	kern_return_t ret = KERN_SUCCESS;
	struct progress_notify_stage_data *data = NULL;

	assert(stage != NULL);
	assert(stage->kos_initialized == false);
	assert(stage->kos_data == NULL);

	stage->kos_data_size = sizeof(struct progress_notify_stage_data);
	ret = kmem_alloc(kernel_map, (vm_offset_t*) &stage->kos_data, stage->kos_data_size,
	    KMA_DATA, VM_KERN_MEMORY_DIAG);
	if (KERN_SUCCESS != ret) {
		printf("progress_notify_stage_initialize failed to allocate memory. Error 0x%x\n", ret);
		return ret;
	}

	data = (struct progress_notify_stage_data *) stage->kos_data;
	data->last_notify_timestamp = 0;
	nanoseconds_to_absolutetime(NOTIFY_INTERVAL_NSECS, &data->notify_interval_matus);

	stage->kos_funcs.kosf_reset = progress_notify_stage_reset;
	stage->kos_funcs.kosf_outproc = progress_notify_stage_outproc;
	stage->kos_funcs.kosf_free = progress_notify_stage_free;

	stage->kos_initialized = true;

	return KERN_SUCCESS;
}

#endif // CONFIG_KDP_INTERACTIVE_DEBUGGING
