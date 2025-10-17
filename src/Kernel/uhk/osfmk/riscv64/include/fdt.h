/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 5, 2025.
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
 * Copyright (c) 2016 Patrick Wildt <patrick@blueri.se>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef __RISCV_FDT_H__
#define __RISCV_FDT_H__

#define _RISCV64_BUS_DMA_PRIVATE
#include <machine/bus.h>

struct fdt_attach_args {
	const char		*fa_name;
	int			 fa_node;
	bus_space_tag_t		 fa_iot;
	bus_dma_tag_t		 fa_dmat;
	struct fdt_reg		*fa_reg;
	int			 fa_nreg;
	uint32_t		*fa_intr;
	int			 fa_nintr;
	int			 fa_acells;
	int			 fa_scells;
};

extern int stdout_node;
extern int stdout_speed;
extern bus_space_tag_t fdt_cons_bs_tag;

void *fdt_find_cons(const char *);

#define fdt_intr_enable riscv_intr_enable
#define fdt_intr_establish riscv_intr_establish_fdt
#define fdt_intr_establish_idx riscv_intr_establish_fdt_idx
#define fdt_intr_establish_idx_cpu riscv_intr_establish_fdt_idx_cpu
#define fdt_intr_establish_imap riscv_intr_establish_fdt_imap
#define fdt_intr_establish_imap_cpu riscv_intr_establish_fdt_imap_cpu
#define fdt_intr_establish_msi riscv_intr_establish_fdt_msi
#define fdt_intr_establish_msi_cpu riscv_intr_establish_fdt_msi_cpu
#define fdt_intr_disable riscv_intr_disable
#define fdt_intr_disestablish riscv_intr_disestablish_fdt
#define fdt_intr_get_parent riscv_intr_get_parent
#define fdt_intr_parent_establish riscv_intr_parent_establish_fdt
#define fdt_intr_parent_disestablish riscv_intr_parent_disestablish_fdt
#define fdt_intr_register riscv_intr_register_fdt

#endif /* __RISCV_FDT_H__ */
