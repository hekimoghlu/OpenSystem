/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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
#include "cap.mdh"
#include "cap.pro"

#ifdef HAVE_CAP_GET_PROC

static int
bin_cap(char *nam, char **argv, UNUSED(Options ops), UNUSED(int func))
{
    int ret = 0;
    cap_t caps;
    if(*argv) {
	unmetafy(*argv, NULL);
	caps = cap_from_text(*argv);
	if(!caps) {
	    zwarnnam(nam, "invalid capability string");
	    return 1;
	}
	if(cap_set_proc(caps)) {
	    zwarnnam(nam, "can't change capabilities: %e", errno);
	    ret = 1;
	}
    } else {
	char *result = NULL;
	ssize_t length;
	caps = cap_get_proc();
	if(caps)
	    result = cap_to_text(caps, &length);
	if(!caps || !result) {
	    zwarnnam(nam, "can't get capabilities: %e", errno);
	    ret = 1;
	} else
	    puts(result);
    }
    cap_free(caps);
    return ret;
}

static int
bin_getcap(char *nam, char **argv, UNUSED(Options ops), UNUSED(int func))
{
    int ret = 0;

    do {
	char *result = NULL;
	ssize_t length;
	cap_t caps;

	caps = cap_get_file(unmetafy(dupstring(*argv), NULL));
	if(caps)
	    result = cap_to_text(caps, &length);
	if (!caps || !result) {
	    zwarnnam(nam, "%s: %e", *argv, errno);
	    ret = 1;
	} else
	    printf("%s %s\n", *argv, result);
	cap_free(caps);
    } while(*++argv);
    return ret;
}

static int
bin_setcap(char *nam, char **argv, UNUSED(Options ops), UNUSED(int func))
{
    cap_t caps;
    int ret = 0;

    unmetafy(*argv, NULL);
    caps = cap_from_text(*argv++);
    if(!caps) {
	zwarnnam(nam, "invalid capability string");
	return 1;
    }

    do {
	if(cap_set_file(unmetafy(dupstring(*argv), NULL), caps)) {
	    zwarnnam(nam, "%s: %e", *argv, errno);
	    ret = 1;
	}
    } while(*++argv);
    cap_free(caps);
    return ret;
}

#else /* !HAVE_CAP_GET_PROC */

# define bin_cap    bin_notavail
# define bin_getcap bin_notavail
# define bin_setcap bin_notavail

#endif /* !HAVE_CAP_GET_PROC */

/* module paraphernalia */

static struct builtin bintab[] = {
    BUILTIN("cap",    0, bin_cap,    0,  1, 0, NULL, NULL),
    BUILTIN("getcap", 0, bin_getcap, 1, -1, 0, NULL, NULL),
    BUILTIN("setcap", 0, bin_setcap, 2, -1, 0, NULL, NULL),
};

static struct features module_features = {
    bintab, sizeof(bintab)/sizeof(*bintab),
    NULL, 0,
    NULL, 0,
    NULL, 0,
    0
};

/**/
int
setup_(UNUSED(Module m))
{
    return 0;
}

/**/
int
features_(Module m, char ***features)
{
    *features = featuresarray(m, &module_features);
    return 0;
}

/**/
int
enables_(Module m, int **enables)
{
    return handlefeatures(m, &module_features, enables);
}

/**/
int
boot_(UNUSED(Module m))
{
    return 0;
}

/**/
int
cleanup_(Module m)
{
    return setfeatureenables(m, &module_features, NULL);
}

/**/
int
finish_(UNUSED(Module m))
{
    return 0;
}
