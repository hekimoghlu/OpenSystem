/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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
 * Copyright 2004 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef	_DT_MODULE_H
#define	_DT_MODULE_H

#include <dt_impl.h>

#ifdef	__cplusplus
extern "C" {
#endif

extern dt_module_t *dt_module_create(dtrace_hdl_t *, const char *);
extern int dt_module_load(dtrace_hdl_t *, dt_module_t *);
extern void dt_module_unload(dtrace_hdl_t *, dt_module_t *);
extern void dt_module_destroy(dtrace_hdl_t *, dt_module_t *);

extern dt_module_t *dt_module_lookup_by_name(dtrace_hdl_t *, const char *);
extern dt_module_t *dt_module_lookup_by_ctf(dtrace_hdl_t *, ctf_file_t *);

extern ctf_file_t *dt_module_getctf(dtrace_hdl_t *, dt_module_t *);
extern dt_ident_t *dt_module_extern(dtrace_hdl_t *, dt_module_t *,
    const char *, const dtrace_typeinfo_t *);

extern const char *dt_module_modelname(dt_module_t *);

#if defined(__APPLE__)
extern void dt_module_get_types(dtrace_hdl_t *,
	const dtrace_probedesc_t *,
	dtrace_argdesc_t *, int *);


extern uint64_t dt_module_sym_location(dt_module_t *, uint8_t, uint64_t);

extern void dtrace_update_kernel_symbols(dtrace_hdl_t *);
#endif /* defined(__APPLE__) */

#ifdef	__cplusplus
}
#endif

#endif	/* _DT_MODULE_H */
