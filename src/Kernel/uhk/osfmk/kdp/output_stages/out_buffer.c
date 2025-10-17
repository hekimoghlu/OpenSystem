/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
#include <vm/vm_kern_xnu.h>

struct buffer_stage_data {
	size_t total_buffer_size;
	size_t current_size;
	char   buffer[];
};

static void
buffer_stage_reset(struct kdp_output_stage *stage)
{
	struct buffer_stage_data *data = (struct buffer_stage_data *) stage->kos_data;

	data->current_size = 0;
	stage->kos_bypass = false;
	stage->kos_bytes_written = 0;
}

static kern_return_t
buffer_stage_flush(struct kdp_output_stage *stage)
{
	kern_return_t err = KERN_SUCCESS;
	struct buffer_stage_data *data = (struct buffer_stage_data *) stage->kos_data;
	struct kdp_output_stage *next_stage = STAILQ_NEXT(stage, kos_next);

	err = next_stage->kos_funcs.kosf_outproc(next_stage, KDP_DATA, NULL, data->current_size, data->buffer);

	if (KERN_SUCCESS != err) {
		return err;
	} else {
		stage->kos_bytes_written += data->current_size;
		data->current_size = 0;
	}

	return err;
}

static kern_return_t
buffer_stage_outproc(struct kdp_output_stage *stage, unsigned int request,
    char *corename, uint64_t length, void * panic_data)
{
	kern_return_t err = KERN_SUCCESS;
	struct buffer_stage_data *data = (struct buffer_stage_data *) stage->kos_data;
	struct kdp_output_stage  *next_stage = STAILQ_NEXT(stage, kos_next);

	boolean_t should_flush = FALSE;

	assert(next_stage != NULL);

	if ((data->current_size && (request == KDP_SEEK || request == KDP_FLUSH || request == KDP_EOF))
	    || (request == KDP_DATA && length == 0 && !panic_data)) {
		should_flush = TRUE;
	}

	if (should_flush) {
		err = buffer_stage_flush(stage);
		if (KERN_SUCCESS != err) {
			kern_coredump_log(NULL, "buffer_stage_outproc (during flush) returned 0x%x\n", err);
			return err;
		}
	}

	if (request == KDP_WRQ || request == KDP_SEEK || request == KDP_EOF) {
		err = next_stage->kos_funcs.kosf_outproc(next_stage, request, corename, length, panic_data);

		if (KERN_SUCCESS != err) {
			kern_coredump_log(NULL, "buffer_stage_outproc (during forwarding) returned 0x%x\n", err);
			return err;
		}
	} else if (request == KDP_DATA) {
		while (length != 0) {
			size_t bytes_to_copy = data->total_buffer_size - data->current_size;

			if (length < bytes_to_copy) {
				/* Safe to cast to size_t here since we just checked that 'length' is less
				 * than a size_t value. */
				bytes_to_copy = (size_t) length;
			}

			bcopy(panic_data, (void *)((uintptr_t)data->buffer + data->current_size), bytes_to_copy);

			data->current_size += bytes_to_copy;
			length -= bytes_to_copy;
			panic_data = (void *) ((uintptr_t) panic_data + bytes_to_copy);

			if (data->current_size == data->total_buffer_size) {
				err = buffer_stage_flush(stage);
				if (KERN_SUCCESS != err) {
					kern_coredump_log(NULL, "buffer_stage_outproc (during flush) returned 0x%x\n", err);
					return err;
				}
			}
		}
	}

	return err;
}

static void
buffer_stage_free(struct kdp_output_stage *stage)
{
	kmem_free(kernel_map, (vm_offset_t) stage->kos_data, stage->kos_data_size);

	stage->kos_data = NULL;
	stage->kos_data_size = 0;
	stage->kos_initialized = false;
}

kern_return_t
buffer_stage_initialize(struct kdp_output_stage *stage, size_t buffer_size)
{
	kern_return_t ret = KERN_SUCCESS;
	struct buffer_stage_data *data = NULL;

	assert(stage != NULL);
	assert(stage->kos_initialized == false);
	assert(stage->kos_data == NULL);
	assert(buffer_size != 0);

	stage->kos_data_size = sizeof(struct buffer_stage_data) + buffer_size;
	ret = kmem_alloc(kernel_map, (vm_offset_t*) &stage->kos_data, stage->kos_data_size,
	    KMA_DATA, VM_KERN_MEMORY_DIAG);
	if (KERN_SUCCESS != ret) {
		printf("buffer_stage_initialize failed to allocate memory. Error 0x%x\n", ret);
		return ret;
	}

	data = (struct buffer_stage_data *) stage->kos_data;
	data->total_buffer_size = buffer_size;
	data->current_size = 0;

	stage->kos_funcs.kosf_reset = buffer_stage_reset;
	stage->kos_funcs.kosf_outproc = buffer_stage_outproc;
	stage->kos_funcs.kosf_free = buffer_stage_free;

	stage->kos_initialized = true;

	return KERN_SUCCESS;
}

#endif /* CONFIG_KDP_INTERACTIVE_DEBUGGING */
