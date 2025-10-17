/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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
#include <sys/types.h>
#include <inttypes.h>
#include <roken.h>
#include "asn1-common.h"
#include "der.h"

#define HEIM_FUZZER_INTERNALS 1
#include "fuzzer.h"

/*
 */

struct context {
    char *interesting;
    size_t size;
    unsigned long permutation;
    unsigned long iteration;
    size_t current;
};

typedef int (*ap)(struct context *, unsigned char *, size_t);

/*
 *
 */

static int
tag_permutate(struct context *context, unsigned char *data, size_t len) 
{
    return 1;
}

static int
size_permutate(struct context *context, unsigned char *data, size_t len) 
{
    return 1;
}
static int
integer_permutate(struct context *context, unsigned char *data, size_t len) 
{
    return 1;
}
static int
os_permutate(struct context *context, unsigned char *data, size_t len) 
{
    return 1;
}
static int
string_permutate(struct context *context, unsigned char *data, size_t len) 
{
    return 1;
}
static int
oid_permutate(struct context *context, unsigned char *data, size_t len) 
{
    return 1;
}

struct perm {
    char type;
    const char *name;
    ap perm;
} permuntations[] = { 
    { 'T', "tag", tag_permutate },
    { 'S', "size", size_permutate },
    { 'd', "integer", integer_permutate },
    { 'o', "octet string", os_permutate },
    { 's', "string", string_permutate },
    { 'd', "oid", oid_permutate }
};

static unsigned long indefinite_form_loop;
static unsigned long indefinite_form_loop_max = 20;

static void
find_interesting(unsigned char *buf, size_t len, char *interesting)
{
    unsigned char *start_buf = buf;

    memset(interesting, 0, len);

    while (len > 0) {
	int ret;
	Der_class cls;
	Der_type type;
	unsigned int tag;
	size_t sz;
	size_t length;
	int end_tag = 0;

	ret = der_get_tag (buf, len, &cls, &type, &tag, &sz);
	if (ret) {
	    warn("der_get_tag failed at offset %lu: %d", (unsigned long)(buf - start_buf), ret);
	    return;
	}
	if (sz > len) {
	    warn("unreasonable length (%u) > %u",
		 (unsigned)sz, (unsigned)len);
	    return;
	}
		  
	memset(interesting, 'T', sz);
	interesting += sz;
	buf += sz;
	len -= sz;

	ret = der_get_length (buf, len, &length, &sz);
	if (ret) {
	    warnx("der_get_tag: %d", ret);
	    return;
	}
	if (sz > len) {
	    warnx("unreasonable tag length (%u) > %u",
		  (unsigned)sz, (unsigned)len);
	    return;
	}

	memset(interesting, 'S', sz);
	interesting += sz;
	buf += sz;
	len -= sz;

	if (length == ASN1_INDEFINITE) {
	    if ((cls == ASN1_C_UNIV && type == PRIM && tag == UT_OctetString) ||
		(cls == ASN1_C_CONTEXT && type == CONS) ||
		(cls == ASN1_C_UNIV && type == CONS && tag == UT_Sequence) ||
		(cls == ASN1_C_UNIV && type == CONS && tag == UT_Set)) {

	    } else {
		fflush(stdout);
		warnx("indef form used on unsupported object");
		return;
	    }
	    end_tag = 1;
	    if (indefinite_form_loop > indefinite_form_loop_max) {
		warnx("indefinite form used recursively more then %lu "
		     "times, aborting", indefinite_form_loop_max);
	    }
	    indefinite_form_loop++;
	    length = len;
	} else if (length > len) {
	    printf("\n");
	    fflush(stdout);
	    warnx("unreasonable inner length (%u) > %u",
		  (unsigned)length, (unsigned)len);
	    return;
	}
	if (cls == ASN1_C_CONTEXT || cls == ASN1_C_APPL) {

	    if (type == CONS) {
		size_t offset = buf - start_buf;
		find_interesting(buf, length, interesting + offset);
	    }
	} else if (cls == ASN1_C_UNIV) {
	    switch (tag) {
	    case UT_EndOfContent:
		break;
	    case UT_Set :
	    case UT_Sequence : {
		find_interesting(buf, length, interesting);
		break;
	    }
	    case UT_Integer :
		memset(interesting, 'i', length);
		break;
	    case UT_OctetString : {
		heim_octet_string str;
		Der_class cls2;
		Der_type type2;
		unsigned int tag2;
		
		ret = der_get_octet_string (buf, length, &str, NULL);
		if (ret) {
		    warnx( "der_get_octet_string: %d", ret);
		    return;
		}

		ret = der_get_tag(str.data, str.length,
				  &cls2, &type2, &tag2, &sz);
		if (ret || sz > str.length || type2 != CONS || tag2 != UT_Sequence) {
		    memset(interesting, 'o', length);
		} else {
		    find_interesting(str.data, str.length, interesting);
		}
		break;
	    }
	    case UT_IA5String :
	    case UT_PrintableString :
	    case UT_GeneralizedTime :
	    case UT_GeneralString :
	    case UT_VisibleString :
	    case UT_UTF8String : {
		memset(interesting, 's', length);
		break;
	    }
	    case UT_OID:
		memset(interesting, 'd', length);
		break;
	    case UT_Enumerated:
		memset(interesting, 'i', length);
		break;
	    default :
		break;
	    }
	}
	if (end_tag) {
	    if (indefinite_form_loop == 0)
		errx(1, "internal error in indefinite form loop detection");
	    indefinite_form_loop--;
	}
	interesting += length;
	buf += length;
	len -= length;
    }
}


/*
 *
 */

static unsigned long
asn1_tries(size_t length)
{
    return length * 12;
}

static int
asn1_fuzz(void **ctx, unsigned long iteration, uint8_t *data, size_t length)
{
    struct context *context;
    int ret;

    if (*ctx == NULL) {
	context = calloc(1, sizeof(*context));

	context->interesting = calloc(1, length + 1);
	if (context->interesting == NULL)
	    abort();
	context->size = length;

	context->interesting[length] = 'e';

	find_interesting(data, length, context->interesting);

	if (context->interesting[length] != 'e') {
	    printf("find_interesting corrupted interesting buffer\n");
	    return 1;
	}

	if (ctx)
	    *ctx = context;

#if 1
	size_t s;
	for (s = 0; s < length; s++) {
	    if (context->interesting[s]) {
		printf("%c", context->interesting[s]);
	    } else {
		printf(".");
	    }
	}
	printf("\n");
#endif

    } else {
	context = *ctx;
    }

    context->iteration = iteration;

    do {
        ret = permuntations[context->permutation].perm(context, data, length);
	if (ret)
	    context->permutation++;

    } while (ret != 0 && context->permutation < sizeof(permuntations)/sizeof(permuntations[0]));

    return ret;
}

static void
asn1_fuzz_free(void *ctx)
{
    struct context *context = (struct context *)ctx;
    free(context->interesting);
    free(context);
}


const struct heim_fuzz_type_data __heim_fuzz_asn1 = {
    "asn1",
    asn1_tries,
    asn1_fuzz,
    asn1_fuzz_free
};

