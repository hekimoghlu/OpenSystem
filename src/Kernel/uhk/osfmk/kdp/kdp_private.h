/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
 * Private functions for kdp.c
 */

static boolean_t
kdp_unknown(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_connect(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_disconnect(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_reattach(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_hostinfo(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_suspend(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_readregs(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_writeregs(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_version(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_kernelversion(
	kdp_pkt_t             *,
	int                   *,
	unsigned short        *
	);

static boolean_t
kdp_regions(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_maxbytes(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_readmem(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_readmem64(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_readphysmem64(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_writemem(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_writemem64(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_writephysmem64(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_resumecpus(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_breakpoint_set(
	kdp_pkt_t *,
	int *,
	unsigned short *t
	);

static boolean_t
kdp_breakpoint64_set(
	kdp_pkt_t *,
	int  *,
	unsigned short *t
	);


static boolean_t
kdp_breakpoint_remove(
	kdp_pkt_t *,
	int *,
	unsigned short *
	);

static boolean_t
kdp_breakpoint64_remove(
	kdp_pkt_t *,
	int   *,
	unsigned short *
	);


static boolean_t
kdp_reboot(
	kdp_pkt_t *,
	int   *,
	unsigned short *
	);

static boolean_t
kdp_readioport(kdp_pkt_t *, int *, unsigned short *);

static boolean_t
kdp_writeioport(kdp_pkt_t *, int *, unsigned short *);

static boolean_t
kdp_readmsr64(kdp_pkt_t *, int *, unsigned short *);

static boolean_t
kdp_writemsr64(kdp_pkt_t *, int *, unsigned short *);

static boolean_t
kdp_dumpinfo(kdp_pkt_t *, int *, unsigned short *);
