/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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
/* $NetBSD: tc_conf.h,v 1.10 2000/06/04 19:14:29 cgd Exp $ */

/*
 * Copyright (c) 1995 Carnegie-Mellon University.
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

/*
 * Machine-specific TurboChannel configuration definitions.
 */

#ifdef DEC_3000_500
#include <alpha/tc/tc_dma_3000_500.h>

extern void	tc_3000_500_intr_setup(void);
extern void	tc_3000_500_iointr(void *, unsigned long);

extern void	tc_3000_500_intr_establish(struct device *, void *,
		    int, int (*)(void *), void *, const char *);
extern void	tc_3000_500_intr_disestablish(struct device *, void *,
		    const char *);

extern int	tc_3000_500_nslots;
extern struct tc_slotdesc tc_3000_500_slots[];
extern int	tc_3000_500_graphics_nbuiltins;
extern struct tc_builtin tc_3000_500_graphics_builtins[];
extern int	tc_3000_500_nographics_nbuiltins;
extern struct tc_builtin tc_3000_500_nographics_builtins[];

extern void	tc_3000_500_ioslot(u_int32_t, u_int32_t, int);

extern int	tc_3000_500_activate(struct device *, int);
#endif /* DEC_3000_500 */

#ifdef DEC_3000_300
#include <alpha/tc/tc_dma_3000_300.h>

extern void	tc_3000_300_intr_setup(void);
extern void	tc_3000_300_iointr(void *, unsigned long);

extern void	tc_3000_300_intr_establish(struct device *, void *,
		    int, int (*)(void *), void *, const char *);
extern void	tc_3000_300_intr_disestablish(struct device *, void *,
		    const char *);

extern int	tc_3000_300_nslots;
extern struct tc_slotdesc tc_3000_300_slots[];
extern int	tc_3000_300_nbuiltins;
extern struct tc_builtin tc_3000_300_builtins[];
#endif /* DEC_3000_300 */

extern int	tc_fb_cnattach(tc_addr_t);
