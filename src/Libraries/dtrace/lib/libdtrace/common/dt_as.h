/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
 * Copyright 2005 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef	_DT_AS_H
#define	_DT_AS_H

#include <sys/types.h>
#include <sys/dtrace.h>

#ifdef	__cplusplus
extern "C" {
#endif

typedef struct dt_irnode {
	uint_t di_label;		/* label number or DT_LBL_NONE */
	dif_instr_t di_instr;		/* instruction opcode */
	void *di_extern;		/* opcode-specific external reference */
	struct dt_irnode *di_next;	/* next instruction */
} dt_irnode_t;

#define	DT_LBL_NONE	0		/* no label on this instruction */

typedef struct dt_irlist {
	dt_irnode_t *dl_list;		/* pointer to first node in list */
	dt_irnode_t *dl_last;		/* pointer to last node in list */
	uint_t dl_len;			/* number of valid instructions */
	uint_t dl_label;		/* next label number to assign */
} dt_irlist_t;

extern void dt_irlist_create(dt_irlist_t *);
extern void dt_irlist_destroy(dt_irlist_t *);
extern void dt_irlist_append(dt_irlist_t *, dt_irnode_t *);
extern uint_t dt_irlist_label(dt_irlist_t *);

#ifdef	__cplusplus
}
#endif

#endif	/* _DT_AS_H */
