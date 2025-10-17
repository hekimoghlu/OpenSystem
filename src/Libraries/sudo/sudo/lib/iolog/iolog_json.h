/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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
#ifndef IOLOG_JSON_H
#define IOLOG_JSON_H

#include "sudo_json.h"
#include "sudo_queue.h"

TAILQ_HEAD(json_item_list, json_item);

struct json_object {
    struct json_item *parent;
    struct json_item_list items;
};

struct json_item {
    TAILQ_ENTRY(json_item) entries;
    char *name;		/* may be NULL for first brace */
    unsigned int lineno;
    enum json_value_type type;
    union {
	struct json_object child;
	char *string;
	long long number;
	id_t id;
	bool boolean;
    } u;
};

void free_json_items(struct json_item_list *items);
bool iolog_parse_json(FILE *fp, const char *filename, struct json_object *root);
char **json_array_to_strvec(struct json_object *array);

#endif /* IOLOG_JSON_H */
