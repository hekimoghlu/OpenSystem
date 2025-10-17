/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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
#include "langinfo.mdh"
#include "langinfo.pro"

#ifdef HAVE_LANGINFO_H
# include <langinfo.h>
#endif

/**/
#ifdef HAVE_NL_LANGINFO

static char *nl_names[] = {
#ifdef CODESET
    "CODESET",
#endif /* CODESET */
#ifdef D_T_FMT
    "D_T_FMT",
#endif /* D_T_FMT */
#ifdef D_FMT
    "D_FMT",
#endif /* D_FMT */
#ifdef T_FMT
    "T_FMT",
#endif /* T_FMT */
#ifdef RADIXCHAR
    "RADIXCHAR",
#endif /* RADIXCHAR */
#ifdef THOUSEP
    "THOUSEP",
#endif /* THOUSEP */
#ifdef YESEXPR
    "YESEXPR",
#endif /* YESEXPR */
#ifdef NOEXPR
    "NOEXPR",
#endif /* NOEXPR */
#ifdef CRNCYSTR
    "CRNCYSTR",
#endif /* CRNCYSTR */
#ifdef ABDAY_1
    "ABDAY_1",
#endif /* ABDAY_1 */
#ifdef ABDAY_2
    "ABDAY_2",
#endif /* ABDAY_2 */
#ifdef ABDAY_3
    "ABDAY_3",
#endif /* ABDAY_3 */
#ifdef ABDAY_4
    "ABDAY_4",
#endif /* ABDAY_4 */
#ifdef ABDAY_5
    "ABDAY_5",
#endif /* ABDAY_5 */
#ifdef ABDAY_6
    "ABDAY_6",
#endif /* ABDAY_6 */
#ifdef ABDAY_7
    "ABDAY_7",
#endif /* ABDAY_7 */
#ifdef DAY_1
    "DAY_1",
#endif /* DAY_1 */
#ifdef DAY_2
    "DAY_2",
#endif /* DAY_2 */
#ifdef DAY_3
    "DAY_3",
#endif /* DAY_3 */
#ifdef DAY_4
    "DAY_4",
#endif /* DAY_4 */
#ifdef DAY_5
    "DAY_5",
#endif /* DAY_5 */
#ifdef DAY_6
    "DAY_6",
#endif /* DAY_6 */
#ifdef DAY_7
    "DAY_7",
#endif /* DAY_7 */
#ifdef ABMON_1
    "ABMON_1",
#endif /* ABMON_1 */
#ifdef ABMON_2
    "ABMON_2",
#endif /* ABMON_2 */
#ifdef ABMON_3
    "ABMON_3",
#endif /* ABMON_3 */
#ifdef ABMON_4
    "ABMON_4",
#endif /* ABMON_4 */
#ifdef ABMON_5
    "ABMON_5",
#endif /* ABMON_5 */
#ifdef ABMON_6
    "ABMON_6",
#endif /* ABMON_6 */
#ifdef ABMON_7
    "ABMON_7",
#endif /* ABMON_7 */
#ifdef ABMON_8
    "ABMON_8",
#endif /* ABMON_8 */
#ifdef ABMON_9
    "ABMON_9",
#endif /* ABMON_9 */
#ifdef ABMON_10
    "ABMON_10",
#endif /* ABMON_10 */
#ifdef ABMON_11
    "ABMON_11",
#endif /* ABMON_11 */
#ifdef ABMON_12
    "ABMON_12",
#endif /* ABMON_12 */
#ifdef MON_1
    "MON_1",
#endif /* MON_1 */
#ifdef MON_2
    "MON_2",
#endif /* MON_2 */
#ifdef MON_3
    "MON_3",
#endif /* MON_3 */
#ifdef MON_4
    "MON_4",
#endif /* MON_4 */
#ifdef MON_5
    "MON_5",
#endif /* MON_5 */
#ifdef MON_6
    "MON_6",
#endif /* MON_6 */
#ifdef MON_7
    "MON_7",
#endif /* MON_7 */
#ifdef MON_8
    "MON_8",
#endif /* MON_8 */
#ifdef MON_9
    "MON_9",
#endif /* MON_9 */
#ifdef MON_10
    "MON_10",
#endif /* MON_10 */
#ifdef MON_11
    "MON_11",
#endif /* MON_11 */
#ifdef MON_12
    "MON_12",
#endif /* MON_12 */
#ifdef T_FMT_AMPM
    "T_FMT_AMPM",
#endif /* T_FMT_AMPM */
#ifdef AM_STR
    "AM_STR",
#endif /* AM_STR */
#ifdef PM_STR
    "PM_STR",
#endif /* PM_STR */
#ifdef ERA
    "ERA",
#endif /* ERA */
#ifdef ERA_D_FMT
    "ERA_D_FMT",
#endif /* ERA_D_FMT */
#ifdef ERA_D_T_FMT
    "ERA_D_T_FMT",
#endif /* ERA_D_T_FMT */
#ifdef ERA_T_FMT
    "ERA_T_FMT",
#endif /* ERA_T_FMT */
#ifdef ALT_DIGITS
    "ALT_DIGITS",
#endif /* ALT_DIGITS */
    NULL
};

static nl_item nl_vals[] = {
#ifdef CODESET
    CODESET,
#endif /* CODESET */
#ifdef D_T_FMT
    D_T_FMT,
#endif /* D_T_FMT */
#ifdef D_FMT
    D_FMT,
#endif /* D_FMT */
#ifdef T_FMT
    T_FMT,
#endif /* T_FMT */
#ifdef RADIXCHAR
    RADIXCHAR,
#endif /* RADIXCHAR */
#ifdef THOUSEP
    THOUSEP,
#endif /* THOUSEP */
#ifdef YESEXPR
    YESEXPR,
#endif /* YESEXPR */
#ifdef NOEXPR
    NOEXPR,
#endif /* NOEXPR */
#ifdef CRNCYSTR
    CRNCYSTR,
#endif /* CRNCYSTR */
#ifdef ABDAY_1
    ABDAY_1,
#endif /* ABDAY_1 */
#ifdef ABDAY_2
    ABDAY_2,
#endif /* ABDAY_2 */
#ifdef ABDAY_3
    ABDAY_3,
#endif /* ABDAY_3 */
#ifdef ABDAY_4
    ABDAY_4,
#endif /* ABDAY_4 */
#ifdef ABDAY_5
    ABDAY_5,
#endif /* ABDAY_5 */
#ifdef ABDAY_6
    ABDAY_6,
#endif /* ABDAY_6 */
#ifdef ABDAY_7
    ABDAY_7,
#endif /* ABDAY_7 */
#ifdef DAY_1
    DAY_1,
#endif /* DAY_1 */
#ifdef DAY_2
    DAY_2,
#endif /* DAY_2 */
#ifdef DAY_3
    DAY_3,
#endif /* DAY_3 */
#ifdef DAY_4
    DAY_4,
#endif /* DAY_4 */
#ifdef DAY_5
    DAY_5,
#endif /* DAY_5 */
#ifdef DAY_6
    DAY_6,
#endif /* DAY_6 */
#ifdef DAY_7
    DAY_7,
#endif /* DAY_7 */
#ifdef ABMON_1
    ABMON_1,
#endif /* ABMON_1 */
#ifdef ABMON_2
    ABMON_2,
#endif /* ABMON_2 */
#ifdef ABMON_3
    ABMON_3,
#endif /* ABMON_3 */
#ifdef ABMON_4
    ABMON_4,
#endif /* ABMON_4 */
#ifdef ABMON_5
    ABMON_5,
#endif /* ABMON_5 */
#ifdef ABMON_6
    ABMON_6,
#endif /* ABMON_6 */
#ifdef ABMON_7
    ABMON_7,
#endif /* ABMON_7 */
#ifdef ABMON_8
    ABMON_8,
#endif /* ABMON_8 */
#ifdef ABMON_9
    ABMON_9,
#endif /* ABMON_9 */
#ifdef ABMON_10
    ABMON_10,
#endif /* ABMON_10 */
#ifdef ABMON_11
    ABMON_11,
#endif /* ABMON_11 */
#ifdef ABMON_12
    ABMON_12,
#endif /* ABMON_12 */
#ifdef MON_1
    MON_1,
#endif /* MON_1 */
#ifdef MON_2
    MON_2,
#endif /* MON_2 */
#ifdef MON_3
    MON_3,
#endif /* MON_3 */
#ifdef MON_4
    MON_4,
#endif /* MON_4 */
#ifdef MON_5
    MON_5,
#endif /* MON_5 */
#ifdef MON_6
    MON_6,
#endif /* MON_6 */
#ifdef MON_7
    MON_7,
#endif /* MON_7 */
#ifdef MON_8
    MON_8,
#endif /* MON_8 */
#ifdef MON_9
    MON_9,
#endif /* MON_9 */
#ifdef MON_10
    MON_10,
#endif /* MON_10 */
#ifdef MON_11
    MON_11,
#endif /* MON_11 */
#ifdef MON_12
    MON_12,
#endif /* MON_12 */
#ifdef T_FMT_AMPM
    T_FMT_AMPM,
#endif /* T_FMT_AMPM */
#ifdef AM_STR
    AM_STR,
#endif /* AM_STR */
#ifdef PM_STR
    PM_STR,
#endif /* PM_STR */
#ifdef ERA
    ERA,
#endif /* ERA */
#ifdef ERA_D_FMT
    ERA_D_FMT,
#endif /* ERA_D_FMT */
#ifdef ERA_D_T_FMT
    ERA_D_T_FMT,
#endif /* ERA_D_T_FMT */
#ifdef ERA_T_FMT
    ERA_T_FMT,
#endif /* ERA_T_FMT */
#ifdef ALT_DIGITS
    ALT_DIGITS,
#endif /* ALT_DIGITS */
    0
};

static nl_item *
liitem(const char *name)
{
    char **element;
    nl_item *nlcode;

    nlcode = &nl_vals[0];

    for (element = (char **)nl_names; *element; element++, nlcode++) {
	if ((!strcmp(*element, name)))
	    return nlcode;
    }

    return NULL;
}

/**/
static HashNode
getlanginfo(UNUSED(HashTable ht), const char *name)
{
    int len;
    nl_item *elem;
    char *listr, *nameu;
    Param pm = NULL;

    nameu = dupstring(name);
    unmetafy(nameu, &len);

    pm = (Param) hcalloc(sizeof(struct param));
    pm->node.nam = nameu;
    pm->node.flags = PM_READONLY | PM_SCALAR;
    pm->gsu.s = &nullsetscalar_gsu;

    if(name)
	elem = liitem(name);
    else
	elem = NULL;

    if (elem && (listr = nl_langinfo(*elem))) {
	pm->u.str = dupstring(listr);
    }
    else
    {
	/* zwarn("no such lang info: %s", name); */
	pm->u.str = dupstring("");
	pm->node.flags |= PM_UNSET;
    }
    return &pm->node;
}

/**/
static void
scanlanginfo(UNUSED(HashTable ht), ScanFunc func, int flags)
{
    Param pm = NULL;
    char **element, *langstr;
    nl_item *nlcode;

    pm = (Param) hcalloc(sizeof(struct param));
    pm->gsu.s = &nullsetscalar_gsu;
    pm->node.flags = PM_READONLY | PM_SCALAR;

    nlcode = &nl_vals[0];
    for (element = (char **)nl_names; *element; element++, nlcode++) {
	if ((langstr = nl_langinfo(*nlcode)) != NULL) {
	    pm->u.str = dupstring(langstr);
	    pm->node.nam = dupstring(*element);
	    func(&pm->node, flags);
	}
    }
    
}

static struct paramdef partab[] = {
    SPECIALPMDEF("langinfo", 0, NULL, getlanginfo, scanlanginfo)
};

/**/
#endif /* HAVE_NL_LANGINFO */

static struct features module_features = {
    NULL, 0,
    NULL, 0,
    NULL, 0,
#ifdef HAVE_NL_LANGINFO
    partab, sizeof(partab)/sizeof(*partab),
#else
    NULL, 0,
#endif
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
