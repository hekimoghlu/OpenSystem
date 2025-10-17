/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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
 * @OSF_COPYRIGHT@
 */
/*
 * Mach Operating System
 * Copyright (c) 1991,1990 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie Mellon
 * the rights to redistribute these changes.
 */

#include <mach_kdp.h>
#include <mach/vm_param.h>
#include <x86_64/lowglobals.h>

/*
 * on x86_64 the low mem vectors live here and get mapped to 0xffffff8000002000 at
 * system startup time
 */

extern void         *version;
extern void         *kmod;
extern void         *kdp_trans_off;
extern void         *kdp_read_io;
extern void         *osversion;
extern void         *flag_kdp_trigger_reboot;
extern void         *manual_pkt;
extern void         *kdp_jtag_coredump;
extern vm_offset_t  c_buffers;
extern vm_size_t    c_buffers_size;

lowglo lowGlo __attribute__ ((aligned(PAGE_SIZE))) = {
	.lgVerCode              = { 'C', 'a', 't', 'f', 'i', 's', 'h', ' ' },

	// Increment major version for changes that break the current usage of lowGlow
	.lgLayoutMajorVersion   = 0,
	// Increment minor version for changes that provide additional fields but do
	// not break the current usage of lowGlow
	.lgLayoutMinorVersion   = 1,

	// Kernel version (not lowglo layout version)
	.lgVersion              = (uint64_t) &version,

	// Kernel compressor buffers
	.lgCompressorBufferAddr = (uint64_t) &c_buffers,
	.lgCompressorSizeAddr   = (uint64_t) &c_buffers_size,

	.lgKmodptr              = (uint64_t) &kmod,

#if MACH_KDP
	.lgTransOff             = (uint64_t) &kdp_trans_off,
	.lgReadIO               = (uint64_t) &kdp_read_io,
#else
	.lgTransOff             = 0,
	.lgReadIO               = 0,
#endif

	.lgDevSlot1             = 0,
	.lgDevSlot2             = 0,

	.lgOSVersion            = (uint64_t) &osversion,

#if MACH_KDP
	.lgRebootFlag           = (uint64_t) &flag_kdp_trigger_reboot,
	.lgManualPktAddr        = (uint64_t) &manual_pkt,
#else
	.lgRebootFlag           = 0,
	.lgManualPktAddr        = 0,
#endif
	.lgKdpJtagCoredumpAddr  = (uint64_t) &kdp_jtag_coredump
};
