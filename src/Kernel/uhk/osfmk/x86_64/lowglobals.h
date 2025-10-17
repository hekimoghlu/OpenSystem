/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
 *		Header files for the Low Memory Globals (lg)
 */
#ifndef _LOW_MEMORY_GLOBALS_H_
#define _LOW_MEMORY_GLOBALS_H_

#include <mach/mach_types.h>
#include <mach/vm_types.h>
#include <mach/machine/vm_types.h>
#include <mach/vm_prot.h>

#ifndef __x86_64__
#error  Wrong architecture - this file is meant for x86_64
#endif

/*
 * Don't change these structures unless you change the corresponding assembly code
 * which is in lowmem_vectors.s
 */

#pragma pack(8)         /* Make sure the structure stays as we defined it */
typedef struct lowglo {
	unsigned char   lgVerCode[8];           /* 0xffffff8000002000 System verification code */
	uint64_t        lgZero;                 /* 0xffffff8000002008 Double constant 0 */
	uint64_t        lgStext;                /* 0xffffff8000002010 Start of kernel text */
	uint64_t        lgLayoutMajorVersion;   /* 0xffffff8000002018 Low globals layout major version */
	uint64_t        lgLayoutMinorVersion;   /* 0xffffff8000002020 Low globals layout minor version */
	uint64_t        lgRsv028;               /* 0xffffff8000002028 Reserved */
	uint64_t        lgVersion;              /* 0xffffff8000002030 Pointer to kernel version string */
	uint64_t        lgCompressorBufferAddr; /* 0xffffff8000002038 Pointer to compressor buffer */
	uint64_t        lgCompressorSizeAddr;   /* 0xffffff8000002040 Pointer to size of compressor buffer */
	uint64_t        lgRsv038[278];          /* 0xffffff8000002048 Reserved */
	uint64_t        lgKmodptr;              /* 0xffffff80000028f8 Pointer to kmod, debugging aid */
	uint64_t        lgTransOff;             /* 0xffffff8000002900 Pointer to kdp_trans_off, debugging aid */
	uint64_t        lgReadIO;               /* 0xffffff8000002908 Pointer to kdp_read_io, debugging aid */
	uint64_t        lgDevSlot1;             /* 0xffffff8000002910 For developer use */
	uint64_t        lgDevSlot2;             /* 0xffffff8000002918 For developer use */
	uint64_t        lgOSVersion;            /* 0xffffff8000002920 Pointer to OS version string */
	uint64_t        lgRebootFlag;           /* 0xffffff8000002928 Pointer to debugger reboot trigger */
	uint64_t        lgManualPktAddr;        /* 0xffffff8000002930 Pointer to manual packet structure */
	uint64_t        lgKdpJtagCoredumpAddr;  /* 0xffffff8000002938 Pointer to kdp_jtag_coredump_t structure */

	uint64_t        lgRsv940[216];          /* 0xffffff8000002940 Reserved - push to 1 page */
} lowglo;
#pragma pack()
extern lowglo lowGlo;
#endif /* _LOW_MEMORY_GLOBALS_H_ */
