/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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
#include "includes.h"

#include <sys/types.h>

#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "misc.h"
#include "servconf.h"
#include "xmalloc.h"
#include "hostfile.h"
#include "auth.h"

extern ServerOptions options;

/*
 * Configuration of enabled authentication methods. Separate from the rest of
 * auth2-*.c because we want to query it during server configuration validity
 * checking in the sshd listener process without pulling all the auth code in
 * too.
 */

/* "none" is allowed only one time and it is cleared by userauth_none() later */
int none_enabled = 1;
struct authmethod_cfg methodcfg_none = {
	"none",
	NULL,
	&none_enabled
};
struct authmethod_cfg methodcfg_pubkey = {
	"publickey",
	"publickey-hostbound-v00@openssh.com",
	&options.pubkey_authentication
};
#ifdef GSSAPI
struct authmethod_cfg methodcfg_gssapi = {
	"gssapi-with-mic",
	NULL,
	&options.gss_authentication
};
#endif
struct authmethod_cfg methodcfg_passwd = {
	"password",
	NULL,
	&options.password_authentication
};
struct authmethod_cfg methodcfg_kbdint = {
	"keyboard-interactive",
	NULL,
	&options.kbd_interactive_authentication
};
struct authmethod_cfg methodcfg_hostbased = {
	"hostbased",
	NULL,
	&options.hostbased_authentication
};

static struct authmethod_cfg *authmethod_cfgs[] = {
	&methodcfg_none,
	&methodcfg_pubkey,
#ifdef GSSAPI
	&methodcfg_gssapi,
#endif
	&methodcfg_passwd,
	&methodcfg_kbdint,
	&methodcfg_hostbased,
	NULL
};

/*
 * Check a comma-separated list of methods for validity. If need_enable is
 * non-zero, then also require that the methods are enabled.
 * Returns 0 on success or -1 if the methods list is invalid.
 */
int
auth2_methods_valid(const char *_methods, int need_enable)
{
	char *methods, *omethods, *method, *p;
	u_int i, found;
	int ret = -1;
	const struct authmethod_cfg *cfg;

	if (*_methods == '\0') {
		error("empty authentication method list");
		return -1;
	}
	omethods = methods = xstrdup(_methods);
	while ((method = strsep(&methods, ",")) != NULL) {
		for (found = i = 0; !found && authmethod_cfgs[i] != NULL; i++) {
			cfg = authmethod_cfgs[i];
			if ((p = strchr(method, ':')) != NULL)
				*p = '\0';
			if (strcmp(method, cfg->name) != 0)
				continue;
			if (need_enable) {
				if (cfg->enabled == NULL ||
				    *(cfg->enabled) == 0) {
					error("Disabled method \"%s\" in "
					    "AuthenticationMethods list \"%s\"",
					    method, _methods);
					goto out;
				}
			}
			found = 1;
			break;
		}
		if (!found) {
			error("Unknown authentication method \"%s\" in list",
			    method);
			goto out;
		}
	}
	ret = 0;
 out:
	free(omethods);
	return ret;
}
