/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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
#include "asn1-common.h"

struct test_case {
    void *val;
    ssize_t byte_len;
    const char *bytes;
    char *name;
};

typedef int (ASN1CALL *generic_encode)(unsigned char *, size_t, void *, size_t *);
typedef int (ASN1CALL *generic_length)(void *);
typedef int (ASN1CALL *generic_decode)(unsigned char *, size_t, void *, size_t *);
typedef int (ASN1CALL *generic_free)(void *);
typedef int (ASN1CALL *generic_copy)(const void *, void *);

int
generic_test (const struct test_case *tests,
	      unsigned ntests,
	      size_t data_size,
	      int (ASN1CALL *encode)(unsigned char *, size_t, void *, size_t *),
	      int (ASN1CALL *length)(void *),
	      int (ASN1CALL *decode)(unsigned char *, size_t, void *, size_t *),
	      int (ASN1CALL *free_data)(void *),
	      int (*cmp)(void *a, void *b),
	      int (ASN1CALL *copy)(const void *a, void *b));

int
generic_decode_fail(const struct test_case *tests,
		    unsigned ntests,
		    size_t data_size,
		    int (ASN1CALL *decode)(unsigned char *, size_t, void *, size_t *));


struct map_page;

enum map_type { OVERRUN, UNDERRUN };

void *	map_alloc(enum map_type, const void *, size_t, struct map_page **);
void	map_free(struct map_page *, const char *, const char *);
