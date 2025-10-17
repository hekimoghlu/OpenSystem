/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
/*
 *  ucode.h
 *
 *  Interface definitions for the microcode updater interface sysctl
 */

/* Intel defined microcode format */
struct intel_ucupdate {
	/* Header information */
	uint32_t header_version;
	uint32_t update_revision;
	uint32_t date;
	uint32_t processor_signature;
	uint32_t checksum;
	uint32_t loader_revision;
	uint32_t processor_flags;
	uint32_t data_size;
	uint32_t total_size;

	/* Reserved for future expansion */
	uint32_t reserved0;
	uint32_t reserved1;
	uint32_t reserved2;

	/* First word of the update data */
	uint32_t data;
};

extern int ucode_interface(uint64_t addr);
extern void ucode_update_wake_and_apply_cpu_was(void);
