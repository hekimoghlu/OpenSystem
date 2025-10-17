/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#include "deltochar.mdh"
#include "deltochar.pro"

static Widget w_deletetochar;
static Widget w_zaptochar;

/**/
static int
deltochar(UNUSED(char **args))
{
    ZLE_INT_T c = getfullchar(0);
    int dest = zlecs, ok = 0, n = zmult;
    int zap = (bindk->widget == w_zaptochar);

    if (n > 0) {
	while (n-- && dest != zlell) {
	    while (dest != zlell && (ZLE_INT_T)zleline[dest] != c)
		INCPOS(dest);
	    if (dest != zlell) {
		if (!zap || n > 0)
		    INCPOS(dest);
		if (!n) {
		    forekill(dest - zlecs, CUT_RAW);
		    ok++;
		}
	    }
	}
    } else {
	/* ignore character cursor is on when scanning backwards */
	if (dest)
	    DECPOS(dest);
	while (n++ && dest != 0) {
	    while (dest != 0 && (ZLE_INT_T)zleline[dest] != c)
		DECPOS(dest);
	    if ((ZLE_INT_T)zleline[dest] == c) {
		if (!n) {
		    /* HERE adjust zap for trailing combining chars */
		    backkill(zlecs - dest - zap, CUT_RAW|CUT_FRONT);
		    ok++;
		}
		if (dest)
		    DECPOS(dest);
	    }
	}
    }
    return !ok;
}


static struct features module_features = {
    NULL, 0,
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
boot_(Module m)
{
    w_deletetochar = addzlefunction("delete-to-char", deltochar,
                                    ZLE_KILL | ZLE_KEEPSUFFIX);
    if (w_deletetochar) {
	w_zaptochar = addzlefunction("zap-to-char", deltochar,
				     ZLE_KILL | ZLE_KEEPSUFFIX);
	if (w_zaptochar)
	    return 0;
	deletezlefunction(w_deletetochar);
    }
    zwarnnam(m->node.nam, "deltochar: name clash when adding ZLE functions");
    return -1;
}

/**/
int
cleanup_(Module m)
{
    deletezlefunction(w_deletetochar);
    deletezlefunction(w_zaptochar);
    return setfeatureenables(m, &module_features, NULL);
}

/**/
int
finish_(UNUSED(Module m))
{
    return 0;
}
