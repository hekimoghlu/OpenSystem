/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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
#include <kdp/output_stages/output_stages.h>
#include <kdp/kdp_core.h>
#include <kdp/processor_core.h>

static void
net_stage_reset(struct kdp_output_stage *stage)
{
	stage->kos_bypass = false;
	stage->kos_bytes_written = 0;
}

static kern_return_t
net_stage_outproc(struct kdp_output_stage *stage, unsigned int request,
    char *corename, uint64_t length, void * data)
{
	kern_return_t err = KERN_SUCCESS;

	assert(STAILQ_NEXT(stage, kos_next) == NULL);

	err = kdp_send_crashdump_data(request, corename, length, data);
	if (KERN_SUCCESS != err) {
		kern_coredump_log(NULL, "kdp_send_crashdump_data returned 0x%x\n", err);
		return err;
	}

	if (KDP_DATA == request) {
		stage->kos_bytes_written += length;
	}

	return err;
}

static void
net_stage_free(struct kdp_output_stage *stage)
{
	stage->kos_initialized = false;
}

kern_return_t
net_stage_initialize(struct kdp_output_stage *stage)
{
	assert(stage != NULL);
	assert(stage->kos_initialized == false);

	stage->kos_funcs.kosf_reset = net_stage_reset;
	stage->kos_funcs.kosf_outproc = net_stage_outproc;
	stage->kos_funcs.kosf_free = net_stage_free;

	stage->kos_initialized = true;

	return KERN_SUCCESS;
}

#endif /* CONFIG_KDP_INTERACTIVE_DEBUGGING */
