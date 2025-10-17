/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <stdlib.h>
#if defined(HAVE_STDINT_H)
# include <stdint.h>
#elif defined(HAVE_INTTYPES_H)
# include <inttypes.h>
#endif
#include <unistd.h>
#include <errno.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_digest.h"

#ifdef HAVE_SHA224UPDATE
# include <sha2.h>
#else
# include "compat/sha2.h"
#endif

static struct digest_function {
    const unsigned int digest_len;
    void (*init)(SHA2_CTX *);
#ifdef SHA2_VOID_PTR
    void (*update)(SHA2_CTX *, const void *, size_t);
    void (*final)(void *, SHA2_CTX *);
#else
    void (*update)(SHA2_CTX *, const unsigned char *, size_t);
    void (*final)(unsigned char *, SHA2_CTX *);
#endif
} digest_functions[] = {
    {
	SHA224_DIGEST_LENGTH,
	SHA224Init,
	SHA224Update,
	SHA224Final
    }, {
	SHA256_DIGEST_LENGTH,
	SHA256Init,
	SHA256Update,
	SHA256Final
    }, {
	SHA384_DIGEST_LENGTH,
	SHA384Init,
	SHA384Update,
	SHA384Final
    }, {
	SHA512_DIGEST_LENGTH,
	SHA512Init,
	SHA512Update,
	SHA512Final
    }, {
	0
    }
};

struct sudo_digest {
    struct digest_function *func;
    SHA2_CTX ctx;
};

struct sudo_digest *
sudo_digest_alloc_v1(int digest_type)
{
    debug_decl(sudo_digest_alloc, SUDO_DEBUG_UTIL);
    struct digest_function *func = NULL;
    struct sudo_digest *dig;
    int i;

    for (i = 0; digest_functions[i].digest_len != 0; i++) {
	if (digest_type == i) {
	    func = &digest_functions[i];
	    break;
	}
    }
    if (func == NULL) {
	errno = EINVAL;
	debug_return_ptr(NULL);
    }

    if ((dig = malloc(sizeof(*dig))) == NULL)
	debug_return_ptr(NULL);
    func->init(&dig->ctx);
    dig->func = func;

    debug_return_ptr(dig);
}

void
sudo_digest_free_v1(struct sudo_digest *dig)
{
    debug_decl(sudo_digest_free, SUDO_DEBUG_UTIL);

    free(dig);

    debug_return;
}

void
sudo_digest_reset_v1(struct sudo_digest *dig)
{
    debug_decl(sudo_digest_reset, SUDO_DEBUG_UTIL);

    dig->func->init(&dig->ctx);

    debug_return;
}

int
sudo_digest_getlen_v1(int digest_type)
{
    debug_decl(sudo_digest_getlen, SUDO_DEBUG_UTIL);
    int i;

    for (i = 0; digest_functions[i].digest_len != 0; i++) {
	if (digest_type == i)
	    debug_return_int(digest_functions[i].digest_len);
    }

    debug_return_int(-1);
}

void
sudo_digest_update_v1(struct sudo_digest *dig, const void *data, size_t len)
{
    debug_decl(sudo_digest_update, SUDO_DEBUG_UTIL);

    dig->func->update(&dig->ctx, data, len);

    debug_return;
}

void
sudo_digest_final_v1(struct sudo_digest *dig, unsigned char *md)
{
    debug_decl(sudo_digest_final, SUDO_DEBUG_UTIL);

    dig->func->final(md, &dig->ctx);

    debug_return;
}
