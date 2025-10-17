/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 27, 2025.
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
#ifndef SUDO_JSON_H
#define SUDO_JSON_H

#include <sys/types.h>	/* for id_t */

#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif

/*
 * JSON values may be of the following types.
 */
enum json_value_type {
    JSON_STRING,
    JSON_ID,
    JSON_NUMBER,
    JSON_OBJECT,
    JSON_ARRAY,
    JSON_BOOL,
    JSON_NULL
};

/*
 * JSON value suitable for printing.
 * Note: this does not support object values.
 */
struct json_value {
    enum json_value_type type;
    union {
	const char *string;
	long long number;
	id_t id;
	bool boolean;
    } u;
};

struct json_container {
    char *buf;
    unsigned int buflen;
    unsigned int bufsize;
    unsigned int indent_level;
    unsigned int indent_increment;
    bool minimal;
    bool memfatal;
    bool need_comma;
    bool quiet;
};

sudo_dso_public bool sudo_json_init_v1(struct json_container *jsonc, int indent, bool minimal, bool memfatal);
sudo_dso_public bool sudo_json_init_v2(struct json_container *jsonc, int indent, bool minimal, bool memfatal, bool quiet);
#define sudo_json_init(_a, _b, _c, _d, _e) sudo_json_init_v2((_a), (_b), (_c), (_d), (_e))

sudo_dso_public void sudo_json_free_v1(struct json_container *jsonc);
#define sudo_json_free(_a) sudo_json_free_v1((_a))

sudo_dso_public bool sudo_json_open_object_v1(struct json_container *jsonc, const char *name);
#define sudo_json_open_object(_a, _b) sudo_json_open_object_v1((_a), (_b))

sudo_dso_public bool sudo_json_close_object_v1(struct json_container *jsonc);
#define sudo_json_close_object(_a) sudo_json_close_object_v1((_a))

sudo_dso_public bool sudo_json_open_array_v1(struct json_container *jsonc, const char *name);
#define sudo_json_open_array(_a, _b) sudo_json_open_array_v1((_a), (_b))

sudo_dso_public bool sudo_json_close_array_v1(struct json_container *jsonc);
#define sudo_json_close_array(_a) sudo_json_close_array_v1((_a))

sudo_dso_public bool sudo_json_add_value_v1(struct json_container *jsonc, const char *name, struct json_value *value);
#define sudo_json_add_value(_a, _b, _c) sudo_json_add_value_v1((_a), (_b), (_c))

sudo_dso_public bool sudo_json_add_value_as_object_v1(struct json_container *jsonc, const char *name, struct json_value *value);
#define sudo_json_add_value_as_object(_a, _b, _c) sudo_json_add_value_as_object_v1((_a), (_b), (_c))

sudo_dso_public char *sudo_json_get_buf_v1(struct json_container *jsonc);
#define sudo_json_get_buf(_a) sudo_json_get_buf_v1((_a))

sudo_dso_public unsigned int sudo_json_get_len_v1(struct json_container *jsonc);
#define sudo_json_get_len(_a) sudo_json_get_len_v1((_a))

#endif /* SUDO_JSON_H */
