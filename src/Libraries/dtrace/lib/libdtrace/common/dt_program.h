/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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
 * Copyright 2006 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef	_DT_PROGRAM_H
#define	_DT_PROGRAM_H

#ifdef	__cplusplus
extern "C" {
#endif

#include <dtrace.h>
#include <dt_list.h>

typedef struct dt_stmt {
	dt_list_t ds_list;	/* list forward/back pointers */
	dtrace_stmtdesc_t *ds_desc; /* pointer to statement description */
} dt_stmt_t;

struct dtrace_prog {
	dt_list_t dp_list;	/* list forward/back pointers */
	dt_list_t dp_stmts;	/* linked list of dt_stmt_t's */
	ulong_t **dp_xrefs;	/* array of translator reference bitmaps */
	uint_t dp_xrefslen;	/* length of dp_xrefs array */
	uint8_t dp_dofversion;	/* DOF version this program requires */
};

extern dtrace_prog_t *dt_program_create(dtrace_hdl_t *);
extern void dt_program_destroy(dtrace_hdl_t *, dtrace_prog_t *);

extern dtrace_ecbdesc_t *dt_ecbdesc_create(dtrace_hdl_t *,
    const dtrace_probedesc_t *);
extern void dt_ecbdesc_release(dtrace_hdl_t *, dtrace_ecbdesc_t *);

#ifdef	__cplusplus
}
#endif

#endif	/* _DT_PROGRAM_H */
