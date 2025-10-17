/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#include <security/pam_appl.h>

#include "openpam_impl.h"

const char *_pam_func_name[PAM_NUM_PRIMITIVES] = {
	"pam_authenticate",
	"pam_setcred",
	"pam_acct_mgmt",
	"pam_open_session",
	"pam_close_session",
	"pam_chauthtok"
};

const char *_pam_sm_func_name[PAM_NUM_PRIMITIVES] = {
	"pam_sm_authenticate",
	"pam_sm_setcred",
	"pam_sm_acct_mgmt",
	"pam_sm_open_session",
	"pam_sm_close_session",
	"pam_sm_chauthtok"
};

/*
 * Locate a matching dynamic or static module.
 */

pam_module_t *
openpam_load_module(const char *path)
{
	pam_module_t *module;

	module = openpam_dynamic(path);
	openpam_log(PAM_LOG_LIBDEBUG, "%s dynamic %s",
	    (module == NULL) ? "no" : "using", path);

#ifdef OPENPAM_STATIC_MODULES
	/* look for a static module */
	if (module == NULL && strchr(path, '/') == NULL) {
		module = openpam_static(path);
		openpam_log(PAM_LOG_LIBDEBUG, "%s static %s",
		    (module == NULL) ? "no" : "using", path);
	}
#endif
	if (module == NULL) {
		openpam_log(PAM_LOG_ERROR, "no %s found", path);
		return (NULL);
	}
	return (module);
}


/*
 * Release a module.
 * XXX highly thread-unsafe
 */

void
openpam_release_module(pam_module_t *module)
{
	if (module == NULL)
		return;
	if (module->dlh == NULL)
		/* static module */
		return;
	dlclose(module->dlh);
	openpam_log(PAM_LOG_LIBDEBUG, "releasing %s", module->path);
	FREE(module->path);
	FREE(module);
}


/*
 * Destroy a chain, freeing all its links and releasing the modules
 * they point to.
 */

static void
openpam_destroy_chain(pam_chain_t *chain)
{
	if (chain == NULL)
		return;
	openpam_destroy_chain(chain->next);
	chain->next = NULL;
	while (chain->optc) {
		--chain->optc;
		FREE(chain->optv[chain->optc]);
	}
	FREE(chain->optv);
	openpam_release_module(chain->module);
	chain->module = NULL;
	FREE(chain);
}


/*
 * Clear the chains and release the modules
 */

void
openpam_clear_chains(pam_chain_t *policy[])
{
	int i;

	for (i = 0; i < PAM_NUM_FACILITIES; ++i) {
		openpam_destroy_chain(policy[i]);
		policy[i] = NULL;
	}
}

/*
 * NOPARSE
 */
