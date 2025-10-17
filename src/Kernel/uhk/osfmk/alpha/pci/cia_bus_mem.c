/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
 * Copyright (c) 1996 Carnegie-Mellon University.
 * All rights reserved.
 *
 * Author: Chris G. Demetriou
 *
 * Permission to use, copy, modify and distribute this software and
 * its documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND
 * FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie the
 * rights to redistribute these changes.
 */

#include <sys/param.h>
#include <sys/systm.h>
#include <sys/malloc.h>
#include <sys/syslog.h>
#include <sys/device.h>
#include <uvm/uvm_extern.h>

#include <machine/bus.h>

#include <alpha/pci/ciareg.h>
#include <alpha/pci/ciavar.h>

#define	CHIP		cia

#define	CHIP_EX_MALLOC_SAFE(v)	(((struct cia_config *)(v))->cc_mallocsafe)
#define	CHIP_D_MEM_EXTENT(v)	(((struct cia_config *)(v))->cc_d_mem_ex)
#define	CHIP_S_MEM_EXTENT(v)	(((struct cia_config *)(v))->cc_s_mem_ex)

/* Dense region 1 */
#define	CHIP_D_MEM_W1_BUS_START(v)	0x00000000UL
#define	CHIP_D_MEM_W1_BUS_END(v)	0xffffffffUL
#define	CHIP_D_MEM_W1_SYS_START(v)	CIA_PCI_DENSE
#define	CHIP_D_MEM_W1_SYS_END(v)	(CIA_PCI_DENSE + 0xffffffffUL)

/* Sparse region 1 */
#define	CHIP_S_MEM_W1_BUS_START(v)					\
	    HAE_MEM_REG1_START(((struct cia_config *)(v))->cc_hae_mem)
#define	CHIP_S_MEM_W1_BUS_END(v)					\
	    (CHIP_S_MEM_W1_BUS_START(v) + HAE_MEM_REG1_MASK)
#define	CHIP_S_MEM_W1_SYS_START(v)					\
	    CIA_PCI_SMEM1
#define	CHIP_S_MEM_W1_SYS_END(v)					\
	    (CIA_PCI_SMEM1 + ((HAE_MEM_REG1_MASK + 1) << 5) - 1)

/* Sparse region 2 */
#define	CHIP_S_MEM_W2_BUS_START(v)					\
	    HAE_MEM_REG2_START(((struct cia_config *)(v))->cc_hae_mem)
#define	CHIP_S_MEM_W2_BUS_END(v)					\
	    (CHIP_S_MEM_W2_BUS_START(v) + HAE_MEM_REG2_MASK)
#define	CHIP_S_MEM_W2_SYS_START(v)					\
	    CIA_PCI_SMEM2
#define	CHIP_S_MEM_W2_SYS_END(v)					\
	    (CIA_PCI_SMEM2 + ((HAE_MEM_REG2_MASK + 1) << 5) - 1)

/* Sparse region 3 */
#define	CHIP_S_MEM_W3_BUS_START(v)					\
	    HAE_MEM_REG3_START(((struct cia_config *)(v))->cc_hae_mem)
#define	CHIP_S_MEM_W3_BUS_END(v)					\
	    (CHIP_S_MEM_W3_BUS_START(v) + HAE_MEM_REG3_MASK)
#define	CHIP_S_MEM_W3_SYS_START(v)					\
	    CIA_PCI_SMEM3
#define	CHIP_S_MEM_W3_SYS_END(v)					\
	    (CIA_PCI_SMEM3 + ((HAE_MEM_REG3_MASK + 1) << 5) - 1)

#include "pci_swiz_bus_mem_chipdep.c"

