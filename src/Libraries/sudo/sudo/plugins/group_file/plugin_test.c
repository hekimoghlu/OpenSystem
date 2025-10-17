/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>

#include "sudo_plugin.h"

sudo_dso_public int main(int argc, char *argv[]);

/*
 * Simple driver to test sudoer group plugins.
 * usage: plugin_test [-p "plugin.so plugin_args ..."] user:group ...
 */

static void *group_handle;
static struct sudoers_group_plugin *group_plugin;

static int
plugin_printf(int msg_type, const char *fmt, ...)
{
    va_list ap;
    FILE *fp;
	    
    switch (msg_type) {
    case SUDO_CONV_INFO_MSG:
	fp = stdout;
	break;
    case SUDO_CONV_ERROR_MSG:
	fp = stderr;
	break;
    default:
	errno = EINVAL;
	return -1;
    }

    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);

    return 0;
}

/*
 * Load the specified plugin and run its init function.
 * Returns -1 if unable to open the plugin, else it returns
 * the value from the plugin's init function.
 */
static int
group_plugin_load(char *plugin_info)
{
    char *args, path[PATH_MAX], savedch;
    char **argv = NULL;
    int rc;

    /*
     * Fill in .so path and split out args (if any).
     */
    if ((args = strpbrk(plugin_info, " \t")) != NULL) {
	savedch = *args;
	*args = '\0';
    }
    if (strlcpy(path, plugin_info, sizeof(path)) >= sizeof(path)) {
	fprintf(stderr, "path too long: %s\n", plugin_info);
	return -1;
    }
    if (args != NULL)
	*args++ = savedch;

    /* Open plugin and map in symbol. */
    group_handle = dlopen(path, RTLD_LAZY);
    if (!group_handle) {
	fprintf(stderr, "unable to dlopen %s: %s\n", path, dlerror());
	return -1;
    }
    group_plugin = dlsym(group_handle, "group_plugin");
    if (group_plugin == NULL) {
	fprintf(stderr, "unable to find symbol \"group_plugin\" in %s\n", path);
	return -1;
    }

    if (SUDO_API_VERSION_GET_MAJOR(group_plugin->version) != GROUP_API_VERSION_MAJOR) {
	fprintf(stderr,
	    "%s: incompatible group plugin major version %u, expected %d\n",
	    path, SUDO_API_VERSION_GET_MAJOR(group_plugin->version),
	    GROUP_API_VERSION_MAJOR);
	return -1;
    }

    /*
     * Split args into a vector if specified.
     */
    if (args != NULL) {
	int ac = 0, wasblank = 1;
	char *cp, *last;

        for (cp = args; *cp != '\0'; cp++) {
            if (isblank((unsigned char)*cp)) {
                wasblank = 1;
            } else if (wasblank) {
                wasblank = 0;
                ac++;
            }
        }
	if (ac != 0) 	{
	    argv = malloc((ac + 1) * sizeof(char *));
	    if (argv == NULL) {
		perror(NULL);
		return -1;
	    }
	    ac = 0;
	    cp = strtok_r(args, " \t", &last);
	    while (cp != NULL) {
		argv[ac++] = cp;
		cp = strtok_r(NULL, " \t", &last);
	    }
	    argv[ac] = NULL;
	}
    }

    rc = (group_plugin->init)(GROUP_API_VERSION, plugin_printf, argv);

    free(argv);

    return rc;
}

static void
group_plugin_unload(void)
{
    (group_plugin->cleanup)();
    dlclose(group_handle);
    group_handle = NULL;
}

static int
group_plugin_query(const char *user, const char *group,
    const struct passwd *pwd)
{
    return (group_plugin->query)(user, group, pwd);
}

static void
usage(void)
{
    fprintf(stderr,
	"usage: plugin_test [-p \"plugin.so plugin_args ...\"] user:group ...\n");
    exit(EXIT_FAILURE);
}

int
main(int argc, char *argv[])
{
    int ch, i, found;
    char *plugin = "group_file.so";
    char *user, *group;
    struct passwd *pwd;

    while ((ch = getopt(argc, argv, "p:")) != -1) {
	switch (ch) {
	case 'p':
	    plugin = optarg;
	    break;
	default:
	    usage();
	}
    }
    argc -= optind;
    argv += optind;

    if (argc < 1)
	usage();

    if (group_plugin_load(plugin) != 1) {
	fprintf(stderr, "unable to load plugin: %s\n", plugin);
	exit(EXIT_FAILURE);
    }

    for (i = 0; argv[i] != NULL; i++) {
	user = argv[i];
	group = strchr(argv[i], ':');
	if (group == NULL)
	    continue;
	*group++ = '\0';
	pwd = getpwnam(user);
	found = group_plugin_query(user, group, pwd);
	printf("user %s %s in group %s\n", user, found ? "is" : "NOT ", group);
    }
    group_plugin_unload();

    exit(EXIT_SUCCESS);
}

