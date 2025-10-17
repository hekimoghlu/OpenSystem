/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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

#ifndef __arm64__
#error  Wrong architecture - this file is meant for arm64
#endif

#define LOWGLO_LAYOUT_MAGIC             0xC0DEC0DE

/*
 * This structure is bound to lowmem_vectors.c. Make sure changes here are
 * reflected there as well.
 */

#pragma pack(8) /* Make sure the structure stays as we defined it */
typedef struct lowglo {
	unsigned char lgVerCode[8];            /* 0xffffff8000002000 System verification code */
	uint64_t      lgZero;                  /* 0xffffff8000002008 Constant 0 */
	uint64_t      lgStext;                 /* 0xffffff8000002010 Start of kernel text */
	uint64_t      lgVersion;               /* 0xffffff8000002018 Pointer to kernel version string */
	uint64_t      lgOSVersion;             /* 0xffffff8000002020 Pointer to OS version string */
	uint64_t      lgKmodptr;               /* 0xffffff8000002028 Pointer to kmod, debugging aid */
	uint64_t      lgTransOff;              /* 0xffffff8000002030 Pointer to kdp_trans_off, debugging aid */
	uint64_t      lgRebootFlag;            /* 0xffffff8000002038 Pointer to debugger reboot trigger */
	uint64_t      lgManualPktAddr;         /* 0xffffff8000002040 Pointer to manual packet structure */
	uint64_t      lgAltDebugger;           /* 0xffffff8000002048 Pointer to reserved space for alternate kernel debugger */
	uint64_t      lgPmapMemQ;              /* 0xffffff8000002050 Pointer to PMAP memory queue */
	uint64_t      lgPmapMemPageOffset;     /* 0xffffff8000002058 Offset of physical page member in vm_page_t or vm_page_with_ppnum_t */
	uint64_t      lgPmapMemChainOffset;    /* 0xffffff8000002060 Offset of listq in vm_page_t or vm_page_with_ppnum_t */
	uint64_t      lgStaticAddr;            /* 0xffffff8000002068 Static allocation address */
	uint64_t      lgStaticSize;            /* 0xffffff8000002070 Static allocation size */
	uint64_t      lgLayoutMajorVersion;    /* 0xffffff8000002078 Lowglo major layout version */
	uint64_t      lgLayoutMagic;           /* 0xffffff8000002080 Magic value evaluated to determine if lgLayoutVersion is valid */
	uint64_t      lgPmapMemStartAddr;      /* 0xffffff8000002088 Pointer to start of vm_page_t array */
	uint64_t      lgPmapMemEndAddr;        /* 0xffffff8000002090 Pointer to end of vm_page_t array */
	uint64_t      lgPmapMemPagesize;       /* 0xffffff8000002098 size of vm_page_t */
	uint64_t      lgPmapMemFromArrayMask;  /* 0xffffff80000020A0 Mask to indicate page is from vm_page_t array */
	uint64_t      lgPmapMemFirstppnum;     /* 0xffffff80000020A8 physical page number of the first vm_page_t in the array */
	uint64_t      lgPmapMemPackedShift;    /* 0xffffff80000020B0 alignment of packed pointer */
	uint64_t      lgPmapMemPackedBaseAddr; /* 0xffffff80000020B8 base address of that packed pointers are relative to */
	uint64_t      lgLayoutMinorVersion;    /* 0xffffff80000020C0 Lowglo minor layout version */
	uint64_t      lgPageShift;             /* 0xffffff80000020C8 number of shifts from page number to size */
	uint64_t      lgVmFirstPhys;           /* 0xffffff80000020D0 First physical address of kernel-managed DRAM (inclusive) */
	uint64_t      lgVmLastPhys;            /* 0xffffff80000020D8 Last physical address of kernel-managed DRAM (exclusive) */
	uint64_t      lgPhysMapBase;           /* 0xffffff80000020E0 First virtual address of the Physical Aperture (inclusive) */
	uint64_t      lgPhysMapEnd;            /* 0xffffff80000020E8 Last virtual address of the Physical Aperture (exclusive) */
	uint64_t      lgPmapIoRangePtr;        /* 0xffffff80000020F0 Pointer to an array of pmap_io_range_t objects obtained from the device tree. */
	uint64_t      lgNumPmapIoRanges;       /* 0xffffff80000020F8 Number of pmap_io_range regions in the array represented by lgPmapIoRangePtr. */
	uint64_t      lgCompressorBufferAddr;  /* 0xFFFFFF8000002100 Pointer to compressor buffer */
	uint64_t      lgCompressorSizeAddr;    /* 0xFFFFFF8000002108 Pointer to size of compressor buffer */
} lowglo;
#pragma pack()

extern lowglo lowGlo;

void patch_low_glo(void);
void patch_low_glo_static_region(uint64_t address, uint64_t size);
void patch_low_glo_vm_page_info(void *, void *, uint32_t);

#endif /* _LOW_MEMORY_GLOBALS_H_ */
