/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/queue.h>

struct kdp_output_stage;

struct kdp_core_out_state {
	STAILQ_HEAD(, kdp_output_stage) kcos_out_stage;
	struct kdp_output_stage *       kcos_encryption_stage;
	bool                            kcos_enforce_encryption;
	uint64_t                        kcos_totalbytes;
	uint64_t                        kcos_bytes_written;
	uint64_t                        kcos_lastpercent;
	kern_return_t                   kcos_error;
};

struct kdp_output_stage_funcs {
	void (*kosf_reset)(struct kdp_output_stage *stage);
	kern_return_t (*kosf_outproc)(struct kdp_output_stage *stage, unsigned int request,
	    char *corename, uint64_t length, void *panic_data);
	void (*kosf_free)(struct kdp_output_stage *stage);
};

struct kdp_output_stage {
	STAILQ_ENTRY(kdp_output_stage) kos_next;
	bool                           kos_initialized;
	struct kdp_core_out_state *    kos_outstate;
	struct kdp_output_stage_funcs  kos_funcs;
	uint64_t                       kos_bytes_written; // bytes written since the last call to reset()
	bool                           kos_bypass;
	void *                         kos_data;
	size_t                         kos_data_size;
};
