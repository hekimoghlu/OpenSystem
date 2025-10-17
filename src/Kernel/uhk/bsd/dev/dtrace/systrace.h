/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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

#ifndef _SYS_SYSTRACE_H
#define _SYS_SYSTRACE_H

#include <sys/dtrace.h>

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct systrace_sysent {
	dtrace_id_t     stsy_entry;
	dtrace_id_t     stsy_return;
	int32_t         (*stsy_underlying)(struct proc *, void *, int *);
	int32_t         stsy_return_type;
} systrace_sysent_t;

extern systrace_sysent_t *systrace_sysent;
extern systrace_sysent_t *systrace_sysent32;

extern void (*systrace_probe)(dtrace_id_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t);
extern void systrace_stub(dtrace_id_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t);

extern int32_t dtrace_systrace_syscall(struct proc *, void *, int *);

extern void dtrace_systrace_syscall_return(unsigned short, int, int *);

#ifdef  __cplusplus
}
#endif

#endif  /* _SYS_SYSTRACE_H */
