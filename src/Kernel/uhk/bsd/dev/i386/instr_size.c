/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 19, 2025.
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
 *//*	  All Rights Reserved	*/

#include <sys/dtrace.h>
#include <sys/dtrace_glue.h>

#include <sys/dis_tables.h>

/*
 * This subsystem (with the minor exception of the instr_size() function) is
 * is called from DTrace probe context.  This imposes several requirements on
 * the implementation:
 *
 * 1. External subsystems and functions may not be referenced.  The one current
 *    exception is for cmn_err, but only to signal the detection of table
 *    errors.  Assuming the tables are correct, no combination of input is to
 *    trigger a cmn_err call.
 *
 * 2. These functions can't be allowed to be traced.  To prevent this,
 *    all functions in the probe path (everything except instr_size()) must
 *    have names that begin with "dtrace_".
 */

typedef enum dis_isize {
	DIS_ISIZE_INSTR,
	DIS_ISIZE_OPERAND
} dis_isize_t;


/*
 * get a byte from instruction stream
 */
static int
dtrace_dis_get_byte(void *p)
{
	int ret;
	uchar_t **instr = p;

	ret = **instr;
	*instr += 1;

	return (ret);
}

/*
 * Returns either the size of a given instruction, in bytes, or the size of that
 * instruction's memory access (if any), depending on the value of `which'.
 * If a programming error in the tables is detected, the system will panic to
 * ease diagnosis.  Invalid instructions will not be flagged.  They will appear
 * to have an instruction size between 1 and the actual size, and will be
 * reported as having no memory impact.
 */
/* ARGSUSED2 */
static __attribute__((noinline)) int
dtrace_dis_isize(uchar_t *instr, dis_isize_t which, model_t model, int *rmindex)
{
	int sz;
	dis86_t	x;
	uint_t mode = SIZE32;

	mode = (model == DATAMODEL_LP64) ? SIZE64 : SIZE32;

	x.d86_data = (void **)&instr;
	x.d86_get_byte = dtrace_dis_get_byte;
	x.d86_check_func = NULL;

	if (dtrace_disx86(&x, mode) != 0)
		return (-1);

	if (which == DIS_ISIZE_INSTR)
		sz = x.d86_len;		/* length of the instruction */
	else
		sz = x.d86_memsize;	/* length of memory operand */

	if (rmindex != NULL)
		*rmindex = x.d86_rmindex;
	return (sz);
}

int
dtrace_instr_size_isa(uchar_t *instr, model_t model, int *rmindex)
{
	return (dtrace_dis_isize(instr, DIS_ISIZE_INSTR, model, rmindex));
}

int
dtrace_instr_size(uchar_t *instr)
{
	return (dtrace_dis_isize(instr, DIS_ISIZE_INSTR, DATAMODEL_NATIVE,
	    NULL));
}

