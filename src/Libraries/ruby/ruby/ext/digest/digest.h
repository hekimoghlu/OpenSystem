/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
#include "ruby.h"

#define RUBY_DIGEST_API_VERSION	3

typedef int (*rb_digest_hash_init_func_t)(void *);
typedef void (*rb_digest_hash_update_func_t)(void *, unsigned char *, size_t);
typedef int (*rb_digest_hash_finish_func_t)(void *, unsigned char *);

typedef struct {
    int api_version;
    size_t digest_len;
    size_t block_len;
    size_t ctx_size;
    rb_digest_hash_init_func_t init_func;
    rb_digest_hash_update_func_t update_func;
    rb_digest_hash_finish_func_t finish_func;
} rb_digest_metadata_t;

#define DEFINE_UPDATE_FUNC_FOR_UINT(name) \
void \
rb_digest_##name##_update(void *ctx, unsigned char *ptr, size_t size) \
{ \
    const unsigned int stride = 16384; \
 \
    for (; size > stride; size -= stride, ptr += stride) { \
	name##_Update(ctx, ptr, stride); \
    } \
    if (size > 0) name##_Update(ctx, ptr, size); \
}

#define DEFINE_FINISH_FUNC_FROM_FINAL(name) \
int \
rb_digest_##name##_finish(void *ctx, unsigned char *ptr) \
{ \
    return name##_Final(ptr, ctx); \
}
