/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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
#include "der_locl.h"
#include <com_err.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <getarg.h>
#include <err.h>

RCSID("$Id$");


static const char *class_names[] = {
    "UNIV",			/* 0 */
    "APPL",			/* 1 */
    "CONTEXT",			/* 2 */
    "PRIVATE"			/* 3 */
};

static const char *type_names[] = {
    "PRIM",			/* 0 */
    "CONS"			/* 1 */
};

static const char *tag_names[] = {
    "EndOfContent",		/* 0 */
    "Boolean",			/* 1 */
    "Integer",			/* 2 */
    "BitString",		/* 3 */
    "OctetString",		/* 4 */
    "Null",			/* 5 */
    "ObjectID",			/* 6 */
    NULL,			/* 7 */
    NULL,			/* 8 */
    NULL,			/* 9 */
    "Enumerated",		/* 10 */
    NULL,			/* 11 */
    NULL,			/* 12 */
    NULL,			/* 13 */
    NULL,			/* 14 */
    NULL,			/* 15 */
    "Sequence",			/* 16 */
    "Set",			/* 17 */
    NULL,			/* 18 */
    "PrintableString",		/* 19 */
    NULL,			/* 20 */
    NULL,			/* 21 */
    "IA5String",		/* 22 */
    "UTCTime",			/* 23 */
    "GeneralizedTime",		/* 24 */
    NULL,			/* 25 */
    "VisibleString",		/* 26 */
    "GeneralString",		/* 27 */
    NULL,			/* 28 */
    NULL,			/* 29 */
    "BMPString"			/* 30 */
};

static int
get_type(const char *name, const char *list[], unsigned len)
{
    unsigned i;
    for (i = 0; i < len; i++)
	if (list[i] && strcasecmp(list[i], name) == 0)
	    return i;
    return -1;
}

#define SIZEOF_ARRAY(a) (sizeof((a))/sizeof((a)[0]))

const char *
der_get_class_name(unsigned num)
{
    if (num >= SIZEOF_ARRAY(class_names))
	return NULL;
    return class_names[num];
}

int
der_get_class_num(const char *name)
{
    return get_type(name, class_names, SIZEOF_ARRAY(class_names));
}

const char *
der_get_type_name(unsigned num)
{
    if (num >= SIZEOF_ARRAY(type_names))
	return NULL;
    return type_names[num];
}

int
der_get_type_num(const char *name)
{
    return get_type(name, type_names, SIZEOF_ARRAY(type_names));
}

const char *
der_get_tag_name(unsigned num)
{
    if (num >= SIZEOF_ARRAY(tag_names))
	return NULL;
    return tag_names[num];
}

int
der_get_tag_num(const char *name)
{
    return get_type(name, tag_names, SIZEOF_ARRAY(tag_names));
}
