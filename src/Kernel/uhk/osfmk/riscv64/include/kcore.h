/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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
/* public domain */

/* Make sure this is larger than DRAM_BLOCKS on all arm-based platforms */
#define	NPHYS_RAM_SEGS	8

typedef struct cpu_kcore_hdr {
	u_int64_t	kernelbase;		/* value of KERNEL_BASE */
	u_int64_t	kerneloffs;		/* offset of kernel in RAM */
	u_int64_t	staticsize;		/* size of contiguous mapping */
	u_int64_t	pmap_kernel_l1;		/* pmap_kernel()->pm_l1 */
	u_int64_t	pmap_kernel_l2;		/* pmap_kernel()->pm_l2 */
	u_int64_t	reserved[11];
	phys_ram_seg_t	ram_segs[NPHYS_RAM_SEGS];
} cpu_kcore_hdr_t;
