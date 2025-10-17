/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 23, 2024.
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
 * @(#)dlfcn.h	1.4 revision of 95/04/25  09:36:52
 * This is an unpublished work copyright (c) 1992 HELIOS Software GmbH
 * 30159 Hannover, Germany
 */

#ifndef __dlfcn_h__
#define __dlfcn_h__

#ifndef _TCL
#include <tcl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Mode flags for the dlopen routine.
 */
#define RTLD_LAZY	1	/* lazy function call binding */
#define RTLD_NOW	2	/* immediate function call binding */
#define RTLD_GLOBAL	0x100	/* allow symbols to be global */

/*
 * To be able to intialize, a library may provide a dl_info structure
 * that contains functions to be called to initialize and terminate.
 */
struct dl_info {
	void (*init) _ANSI_ARGS_((void));
	void (*fini) _ANSI_ARGS_((void));
};

VOID *dlopen _ANSI_ARGS_((const char *path, int mode));
VOID *dlsym _ANSI_ARGS_((void *handle, const char *symbol));
char *dlerror _ANSI_ARGS_((void));
int dlclose _ANSI_ARGS_((void *handle));

#ifdef __cplusplus
}
#endif

#endif /* __dlfcn_h__ */
