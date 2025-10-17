/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dispatch/dispatch.h>
#include <mach-o/dyld_priv.h>
#include <security/pam_appl.h>
#include <System/sys/codesign.h>

#include "openpam_impl.h"

#ifndef RTLD_NOW
#define RTLD_NOW RTLD_LAZY
#endif


/*
 * OpenPAM internal
 *
 * Attempt to dynamically load a module.
 * Disable LV on the process if necessary.
 *
 * Return values
 * Pointer		errno
 * NULL		==> populated with underlying cause.
 * valid 	==>	0
 */

static void *
openpam_dlopen(const char *path, int mode)
{
	/* Fast path: dyld shared cache. */
	if (_dyld_shared_cache_contains_path(path)) {
		errno = 0;
		return dlopen(path, mode);
	}

	/* Slow path: check file on disk. */
	if (faccessat(AT_FDCWD, path, R_OK, AT_EACCESS) != 0) {
		/* errno populated by faccessat() and returned to caller. */
		return NULL;
	}

	void *dlh = dlopen(path, mode);
	if (dlh != NULL) {
		errno = 0;
		return dlh;
	}

	/*
	 * The module exists and is readable, but failed to load.
	 * If library validation is enabled, try disabling it and then try again.
	 */
	int   csflags = 0;
	pid_t pid     = getpid();
	csops(pid, CS_OPS_STATUS, &csflags, sizeof(csflags));
	if ((csflags & CS_INSTALLER) != 0) {
		/* Installers cannot disable LV (rdar://99454346). */
		errno = ENOTRECOVERABLE;
		return NULL;
	}
	if ((csflags & (CS_FORCED_LV | CS_REQUIRE_LV)) == 0) {
		/* LV is already disabled. */
		errno = ECANCELED;
		return NULL;
	}
	int rv = csops(getpid(), CS_OPS_CLEAR_LV, NULL, 0);
	if (rv != 0) {
		openpam_log(PAM_LOG_ERROR, "csops(CS_OPS_CLEAR_LV) failed: %d", rv);
		errno = ENOTSUP;
		return NULL;
	}
	
	dlh = dlopen(path, mode);
	if (dlh == NULL) {
		/* Failed to load even with LV disabled: re-enable LV. */
		csflags = CS_REQUIRE_LV;
		csops(pid, CS_OPS_SET_STATUS, &csflags, sizeof(csflags));
		errno = EINVAL;
	}

	errno = 0;
	return dlh;
}


/*
 * OpenPAM internal
 *
 * Attempt to load a specific module.
 * On success, populate the `pam_module_t` structure provided.
 *
 * Return values
 * bool			errno
 * false	==> populated with underlying cause.
 * true		==>	0
 
 */

static bool
openpam_dynamic_load_single(const char *path, pam_module_t *module)
{
	void *dlh = openpam_dlopen(path, RTLD_NOW);
	if (dlh != NULL) {
		openpam_log(PAM_LOG_LIBDEBUG, "%s", path);
		module->path = strdup(path);
		if (module->path == NULL)
			goto bail;
		module->dlh = dlh;
		for (int i = 0; i < PAM_NUM_PRIMITIVES; ++i) {
			module->func[i] = (pam_func_t)dlsym(dlh, _pam_sm_func_name[i]);
			if (module->func[i] == NULL)
				openpam_log(PAM_LOG_LIBDEBUG, "%s: %s(): %s", path, _pam_sm_func_name[i], dlerror());
		}
		return true;
	}

bail:
	if (dlh != NULL)
		dlclose(dlh);

	return false;
}


/*
 * OpenPAM internal
 *
 * Locate a dynamically linked module.
 * Prefer a module matching the current major version, otherwise fall back to the unversioned one.
 */

static bool
openpam_dynamic_load(const char *prefix, const char *path, pam_module_t *module)
{
	char *vpath = NULL;
	bool loaded = false;

	/* Try versioned module first. */
	if (asprintf(&vpath, "%s%s.%d", prefix, path, LIB_MAJ) < 0)
		return false;

	loaded = openpam_dynamic_load_single(vpath, module);
	if (!loaded) {
		/* dlopen() + LV disable failure in installer? */
		if (errno == ENOTRECOVERABLE)
			goto bail;
	
		/* Try again with unversioned module: remove LIB_MAJ. */
		*strrchr(vpath, '.') = '\0';
		loaded = openpam_dynamic_load_single(vpath, module);
	}

bail:
	FREE(vpath);
	return loaded;
}


pam_module_t *
openpam_dynamic(const char *path)
{
	pam_module_t *module;

	if ((module = calloc(1, sizeof *module)) == NULL) {
		openpam_log(PAM_LOG_ERROR, "%m");
		goto no_module;
	}

	/* Prepend the standard prefix if not an absolute pathname. */
	if (path[0] != '/') {
		// <rdar://problem/21545156> Add "/usr/local/lib/pam" to the search list
		static dispatch_once_t onceToken;
		static char *pam_modules_dirs  = NULL;
		static char **pam_search_paths = NULL;

		dispatch_once(&onceToken, ^{
			size_t len = strlen(OPENPAM_MODULES_DIR);
			char *tok, *str;
			const char *delim = ";";
			const char sep = delim[0];
			int i, n;

			str = OPENPAM_MODULES_DIR;
			assert(len > 0);
			assert(str[0]     != sep);		/* OPENPAM_MODULES should not start with a ';' */
			assert(str[len-1] != sep);		/* no terminating ';' */
			for (i = 0, n = 1; i < len; i++) n += (str[i] == sep);

			if ((pam_modules_dirs = strdup(OPENPAM_MODULES_DIR)) != NULL &&
				(pam_search_paths = (char **) malloc((n + 1) * sizeof(char *))) != NULL) {
				for (tok = str = pam_modules_dirs, i = 0; i < n; i++)
					pam_search_paths[i] = tok = strsep(&str, delim);
				pam_search_paths[n] = NULL;
			} else {
				openpam_log(PAM_LOG_ERROR, "%m - PAM module search paths won't work!");
			}
		});

		if (pam_search_paths != NULL) {
			int i;
			for (i = 0; pam_search_paths[i] != NULL; i++) {
				if (openpam_dynamic_load(pam_search_paths[i], path, module))
					return module;
				if (errno == ENOTRECOVERABLE)
					goto no_module;
			}
		}
	} else {
		if (openpam_dynamic_load("", path, module))
			return module;
	}

no_module:
	FREE(module);
	return NULL;
}

/*
 * NOPARSE
 */
