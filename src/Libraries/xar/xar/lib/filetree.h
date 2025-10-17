/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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
 * 03-Apr-2005
 * DRI: Rob Braun <bbraun@synack.net>
 */
/*
 * Portions Copyright 2006, Apple Computer, Inc.
 * Christopher Ryan <ryanc@apple.com>
*/

#ifndef _XAR_FILETREE_H_
#define _XAR_FILETREE_H_

#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include <libxml/xmlwriter.h>
#include <libxml/xmlreader.h>

struct __xar_attr_t {
	const char *key;
	const char *value;
	const char *ns;
	const struct __xar_attr_t *next;
};
typedef const struct __xar_attr_t *xar_attr_t;

struct __xar_prop_t {
        const char *key;
        const char *value;
        const struct __xar_prop_t *parent;
        const struct __xar_prop_t *children;
        const struct __xar_prop_t *next;
        const struct __xar_attr_t *attrs;
        const struct __xar_file_t *file;
	const char *prefix;
	const char *ns;
};
typedef const struct __xar_prop_t *xar_prop_t;

#include "ea.h"

struct __xar_file_t {
	const struct __xar_prop_t *props;
	const struct __xar_attr_t *attrs;
	const char *prefix;
	const char *ns;
	const char *fspath;
	char parent_extracted;
	const struct __xar_file_t *parent;
	const struct __xar_file_t *children;
	const struct __xar_file_t *next;
	xar_ea_t eas;
	uint64_t nexteaid;
};

/* Overview:
 * xar_file_t's exist within a xar_archive_t.  xar_prop_t's exist
 * within xar_file_t's and xar_attr_t's exist within xar_prop_t's
 * and xar_file_t's.
 * Basically, a xar_file_t is a container for xar_prop_t's.
 * xar_attr_t's are things like: <foo bar=5>blah</foo>
 * In this example, foo is the key of a xar_prop_t, and blah is
 * the value.  bar is the key of a xar_attr_t which is part of
 * foo's xar_prop_t, and 5 is bar's value.
 * xar_file_t's have xar_attr_t's for the case of:
 * <file id=42>
 * The file has an attribute of "id" with a value of "42".
 */

struct __xar_iter_t {
	const void *iter;
	char *path;
	void *node;
	int nochild;
};



/* Convenience macros for dereferencing the structs */
#define XAR_ATTR(x) ((struct __xar_attr_t *)(x))
#define XAR_FILE(x) ((struct __xar_file_t *)(x))
#define XAR_PROP(x) ((struct __xar_prop_t *)(x))
#define XAR_ITER(x) ((struct __xar_iter_t *)(x))


void xar_file_free(xar_file_t f);
xar_attr_t xar_attr_new(void);
int32_t xar_attr_set(xar_file_t f, const char *prop, const char *key, const char *value);
int32_t xar_attr_pset(xar_file_t f, xar_prop_t p, const char *key, const char *value);
const char *xar_attr_get(xar_file_t f, const char *prop, const char *key);
const char *xar_attr_pget(xar_file_t f, xar_prop_t p, const char *key);
int xar_attr_equals_attr(xar_attr_t a1, xar_attr_t a2);
int xar_attr_equals_attr_ignoring_keys(xar_attr_t a1, xar_attr_t a2, uint64_t key_count, char** keys_to_ignore);
void xar_attr_free(xar_attr_t a);
void xar_file_serialize(xar_file_t f, xmlTextWriterPtr writer);
int xar_prop_serializable(xar_prop_t p);
xar_file_t xar_file_unserialize(xar_t x, xar_file_t parent, xmlTextReaderPtr reader);
xar_file_t xar_file_find(xar_file_t f, const char *path);
xar_file_t xar_file_new(const char *name);
xar_file_t xar_file_new_from_parent(xar_file_t parent, const char *name);
xar_file_t xar_file_replicate(xar_file_t original, xar_file_t newparent);
int xar_file_equals_file(xar_file_t f1, xar_file_t f2);
void xar_file_free(xar_file_t f);

void xar_prop_serialize(xar_prop_t p, xmlTextWriterPtr writer);
int32_t xar_prop_unserialize(xar_file_t f, xar_prop_t parent, xmlTextReaderPtr reader);
void xar_prop_free(xar_prop_t p);
xar_prop_t xar_prop_new(xar_file_t f, xar_prop_t parent);
xar_prop_t xar_prop_pset(xar_file_t f, xar_prop_t p, const char *key, const char *value);
xar_prop_t xar_prop_find(xar_prop_t p, const char *key);
xar_prop_t xar_prop_pget(xar_prop_t p, const char *key);
const char *xar_prop_getkey(xar_prop_t p);
const char *xar_prop_getvalue(xar_prop_t p);
int32_t xar_prop_setkey(xar_prop_t p, const char *key);
int32_t xar_prop_setvalue(xar_prop_t p, const char *value);
xar_prop_t xar_prop_pfirst(xar_file_t f);
xar_prop_t xar_prop_pnext(xar_prop_t p);
void xar_prop_punset(xar_file_t f, xar_prop_t p);
int xar_prop_equals_prop(xar_prop_t prop1, xar_prop_t prop2);

#endif /* _XAR_FILETREE_H_ */
