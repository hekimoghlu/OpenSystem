/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
#include "zleparameter.mdh"
#include "zleparameter.pro"

/* Functions for the zlewidgets special parameter. */

/**/
static char *
widgetstr(Widget w)
{
    if (!w)
	return dupstring("undefined");
    if (w->flags & WIDGET_INT)
	return dupstring("builtin");
    if (w->flags & WIDGET_NCOMP) {
	char *t = (char *) zhalloc(13 + strlen(w->u.comp.wid) +
				   strlen(w->u.comp.func));

	strcpy(t, "completion:");
	strcat(t, w->u.comp.wid);
	strcat(t, ":");
	strcat(t, w->u.comp.func);

	return t;
    }
    return dyncat("user:", w->u.fnnam);
}

/**/
static HashNode
getpmwidgets(UNUSED(HashTable ht), const char *name)
{
    Param pm = NULL;
    Thingy th;

    pm = (Param) hcalloc(sizeof(struct param));
    pm->node.nam = dupstring(name);
    pm->node.flags = PM_SCALAR | PM_READONLY;
    pm->gsu.s = &nullsetscalar_gsu;

    if ((th = (Thingy) thingytab->getnode(thingytab, name)) &&
	!(th->flags & DISABLED))
	pm->u.str = widgetstr(th->widget);
    else {
	pm->u.str = dupstring("");
	pm->node.flags |= PM_UNSET;
    }
    return &pm->node;
}

/**/
static void
scanpmwidgets(UNUSED(HashTable ht), ScanFunc func, int flags)
{
    struct param pm;
    int i;
    HashNode hn;

    memset((void *)&pm, 0, sizeof(struct param));
    pm.node.flags = PM_SCALAR | PM_READONLY;
    pm.gsu.s = &nullsetscalar_gsu;

    for (i = 0; i < thingytab->hsize; i++)
	for (hn = thingytab->nodes[i]; hn; hn = hn->next) {
	    pm.node.nam = hn->nam;
	    if (func != scancountparams &&
		((flags & (SCANPM_WANTVALS|SCANPM_MATCHVAL)) ||
		 !(flags & SCANPM_WANTKEYS)))
		pm.u.str = widgetstr(((Thingy) hn)->widget);
	    func(&pm.node, flags);
	}
}

/* Functions for the zlekeymaps special parameter. */

static char **
keymapsgetfn(UNUSED(Param pm))
{
    int i;
    HashNode hn;
    char **ret, **p;

    p = ret = (char **) zhalloc((keymapnamtab->ct + 1) * sizeof(char *));

    for (i = 0; i < keymapnamtab->hsize; i++)
	for (hn = keymapnamtab->nodes[i]; hn; hn = hn->next)
	    *p++ = dupstring(hn->nam);
    *p = NULL;

    return ret;
}

/*
 * This is a duplicate of stdhash_gsu.  On some systems
 * (such as Cygwin) we can't put a pointer to an imported variable
 * in a compile-time initialiser, so we use this instead.
 */
static const struct gsu_hash zlestdhash_gsu =
{ hashgetfn, hashsetfn, stdunsetfn };
static const struct gsu_array keymaps_gsu =
{ keymapsgetfn, arrsetfn, stdunsetfn };

static struct paramdef partab[] = {
    SPECIALPMDEF("keymaps", PM_ARRAY|PM_READONLY, &keymaps_gsu, NULL, NULL),
    SPECIALPMDEF("widgets", PM_READONLY,
		 &zlestdhash_gsu, getpmwidgets, scanpmwidgets)
};

static struct features module_features = {
    NULL, 0,
    NULL, 0,
    NULL, 0,
    partab, sizeof(partab)/sizeof(*partab),
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
