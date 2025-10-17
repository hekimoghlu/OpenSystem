/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
  * System libraries.
  */
#include "sys_defs.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#ifdef USE_DYNAMIC_MAPS
#if defined(HAS_DLOPEN)
#include <dlfcn.h>
#elif defined(HAS_SHL_LOAD)
#include <dl.h>
#else
#error "USE_DYNAMIC_LIBS requires HAS_DLOPEN or HAS_SHL_LOAD"
#endif

 /*
  * Utility library.
  */
#include <msg.h>
#include <load_lib.h>

/* load_library_symbols - load shared library and look up symbols */

void    load_library_symbols(const char *libname, LIB_FN *libfuncs,
			             LIB_DP *libdata)
{
    static const char myname[] = "load_library_symbols";
    LIB_FN *fn;
    LIB_DP *dp;

#if defined(HAS_DLOPEN)
    void   *handle;
    char   *emsg;

    /*
     * XXX This is basically how FreeBSD dlfunc() silences a compiler warning
     * about a data/function pointer conversion. The solution below is non-
     * portable: it assumes that both data and function pointers are the same
     * in size, and that both have the same representation.
     */
    union {
	void   *dptr;			/* data pointer */
	void    (*fptr) (void);		/* function pointer */
    }       non_portable_union;

    if ((handle = dlopen(libname, RTLD_NOW)) == 0) {
	emsg = dlerror();
	msg_fatal("%s: dlopen failure loading %s: %s", myname, libname,
		  emsg ? emsg : "don't know why");
    }
    if (libfuncs) {
	for (fn = libfuncs; fn->name; fn++) {
	    if ((non_portable_union.dptr = dlsym(handle, fn->name)) == 0) {
		emsg = dlerror();
		msg_fatal("%s: dlsym failure looking up %s in %s: %s", myname,
			  fn->name, libname, emsg ? emsg : "don't know why");
	    }
	    fn->fptr = non_portable_union.fptr;
	    if (msg_verbose > 1)
		msg_info("loaded %s = %p", fn->name, non_portable_union.dptr);
	}
    }
    if (libdata) {
	for (dp = libdata; dp->name; dp++) {
	    if ((dp->dptr = dlsym(handle, dp->name)) == 0) {
		emsg = dlerror();
		msg_fatal("%s: dlsym failure looking up %s in %s: %s", myname,
			  dp->name, libname, emsg ? emsg : "don't know why");
	    }
	    if (msg_verbose > 1)
		msg_info("loaded %s = %p", dp->name, dp->dptr);
	}
    }
#elif defined(HAS_SHL_LOAD)
    shl_t   handle;

    handle = shl_load(libname, BIND_IMMEDIATE, 0);

    if (libfuncs) {
	for (fn = libfuncs; fn->name; fn++) {
	    if (shl_findsym(&handle, fn->name, TYPE_PROCEDURE, &fn->fptr) != 0)
		msg_fatal("%s: shl_findsym failure looking up %s in %s: %m",
			  myname, fn->name, libname);
	    if (msg_verbose > 1)
		msg_info("loaded %s = %p", fn->name, (void *) fn->fptr);
	}
    }
    if (libdata) {
	for (dp = libdata; dp->name; dp++) {
	    if (shl_findsym(&handle, dp->name, TYPE_DATA, &dp->dptr) != 0)
		msg_fatal("%s: shl_findsym failure looking up %s in %s: %m",
			  myname, dp->name, libname);
	    if (msg_verbose > 1)
		msg_info("loaded %s = %p", dp->name, dp->dptr);
	}
    }
#endif
}

#endif
