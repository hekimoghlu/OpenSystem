/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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

#ifndef	_DT_DOF_H
#define	_DT_DOF_H

#include <dtrace.h>

#ifdef	__cplusplus
extern "C" {
#endif

#include <dt_buf.h>

typedef struct dt_dof {
	dtrace_hdl_t *ddo_hdl;		/* libdtrace handle */
	dtrace_prog_t *ddo_pgp;		/* current program */
	uint_t ddo_nsecs;		/* number of sections */
	dof_secidx_t ddo_strsec; 	/* global strings section index */
	dof_secidx_t *ddo_xlimport;	/* imported xlator section indices */
	dof_secidx_t *ddo_xlexport;	/* exported xlator section indices */
	dt_buf_t ddo_secs;		/* section headers */
	dt_buf_t ddo_strs;		/* global strings */
	dt_buf_t ddo_ldata;		/* loadable section data */
	dt_buf_t ddo_udata;		/* unloadable section data */
	dt_buf_t ddo_probes;		/* probe section data */
	dt_buf_t ddo_args;		/* probe arguments section data */
	dt_buf_t ddo_offs;		/* probe offsets section data */
	dt_buf_t ddo_enoffs;		/* is-enabled offsets section data */
	dt_buf_t ddo_rels;		/* probe relocation section data */
	dt_buf_t ddo_xlms;		/* xlate members section data */
} dt_dof_t;

extern void dt_dof_init(dtrace_hdl_t *);
extern void dt_dof_fini(dtrace_hdl_t *);

#ifdef	__cplusplus
}
#endif

#endif	/* _DT_DOF_H */
