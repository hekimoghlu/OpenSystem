/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 28, 2024.
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
typedef void (*heim_type_init)(void *);
typedef heim_object_t (*heim_type_copy)(void *);
typedef int (*heim_type_cmp)(void *, void *);
typedef unsigned long (*heim_type_hash)(void *);

typedef struct heim_type_data *heim_type_t;

enum {
    HEIM_TID_NUMBER = 0,
    HEIM_TID_NULL = 1,
    HEIM_TID_BOOL = 2,
    HEIM_TID_TAGGED_UNUSED2 = 3,
    HEIM_TID_TAGGED_UNUSED3 = 4,
    HEIM_TID_TAGGED_UNUSED4 = 5,
    HEIM_TID_TAGGED_UNUSED5 = 6,
    HEIM_TID_TAGGED_UNUSED6 = 7,
    HEIM_TID_MEMORY = 128,
    HEIM_TID_ARRAY = 129,
    HEIM_TID_DICT = 130,
    HEIM_TID_STRING = 131,
    HEIM_TID_AUTORELEASE = 132,
    HEIM_TID_DATA = 133,
    HEIM_TID_ERROR = 134,
    HEIM_TID_USER = 255

};

struct heim_type_data {
    heim_tid_t tid;
    const char *name;
    heim_type_init init;
    heim_type_dealloc dealloc;
    heim_type_copy copy;
    heim_type_cmp cmp;
    heim_type_hash hash;
};

heim_type_t _heim_get_isa(heim_object_t);

heim_type_t
_heim_create_type(const char *name,
		  heim_type_init init,
		  heim_type_dealloc dealloc,
		  heim_type_copy copy,
		  heim_type_cmp cmp,
		  heim_type_hash hash);

heim_object_t
_heim_alloc_object(heim_type_t type, size_t size);

heim_tid_t
_heim_type_get_tid(heim_type_t type);

/* tagged tid */
extern struct heim_type_data _heim_null_object;
extern struct heim_type_data _heim_bool_object;
extern struct heim_type_data _heim_number_object;
extern struct heim_type_data _heim_string_object;

#ifdef __APPLE__
#define __heim_string_constant(x) ((heim_object_t)CFSTR(x))
#endif
