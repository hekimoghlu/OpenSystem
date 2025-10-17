/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#ifndef _FBT_H
#define _FBT_H

#if defined (__x86_64__)
typedef uint8_t machine_inst_t;
#elif defined(__arm__)
typedef uint16_t machine_inst_t;
#elif defined(__arm64__)
typedef uint32_t machine_inst_t;
#else
#error Unknown Architecture
#endif

#define MAX_FBTP_NAME_CHARS 16

typedef struct fbt_probe {
	struct fbt_probe *fbtp_hashnext;
	machine_inst_t	*fbtp_patchpoint;
	int8_t			fbtp_rval;
	machine_inst_t	fbtp_patchval;
	machine_inst_t	fbtp_savedval;
        machine_inst_t  fbtp_currentval;
	uintptr_t		fbtp_roffset;
	dtrace_id_t		fbtp_id;
	/* FIXME!
	 * This field appears to only be used in error messages.
	 * It puts this structure into the next size bucket in kmem_alloc
	 * wasting 32 bytes per probe. (in i386 only)
	 */
	char			fbtp_name[MAX_FBTP_NAME_CHARS];
	struct modctl	*fbtp_ctl;
	int		fbtp_loadcnt;
	struct fbt_probe *fbtp_next;
} fbt_probe_t;

extern int dtrace_invop(uintptr_t, uintptr_t *, uintptr_t);
extern int fbt_invop(uintptr_t, uintptr_t *, uintptr_t);
extern void fbt_provide_module(void *, struct modctl *);
extern int fbt_enable (void *arg, dtrace_id_t id, void *parg);

extern bool fbt_module_excluded(struct modctl*);
extern bool fbt_excluded(const char *);

extern void fbt_blacklist_init(void);
extern void fbt_provide_probe(struct modctl *ctl, const char *modname, const char *name, machine_inst_t *instr, machine_inst_t *limit);
#endif /* _FBT_H */
