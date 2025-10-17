/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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
#include "config.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dav1d/data.h"

#include "common/attributes.h"
#include "common/validate.h"

#include "src/data.h"
#include "src/ref.h"

uint8_t *dav1d_data_create_internal(Dav1dData *const buf, const size_t sz) {
    validate_input_or_ret(buf != NULL, NULL);

    if (sz > SIZE_MAX / 2) return NULL;
    buf->ref = dav1d_ref_create(sz);
    if (!buf->ref) return NULL;
    buf->data = buf->ref->const_data;
    buf->sz = buf->m.size = sz;
    dav1d_data_props_set_defaults(&buf->m);

    return buf->ref->data;
}

int dav1d_data_wrap_internal(Dav1dData *const buf, const uint8_t *const ptr,
                             const size_t sz,
                             void (*const free_callback)(const uint8_t *data,
                                                         void *cookie),
                             void *const cookie)
{
    validate_input_or_ret(buf != NULL, DAV1D_ERR(EINVAL));
    validate_input_or_ret(ptr != NULL, DAV1D_ERR(EINVAL));
    validate_input_or_ret(free_callback != NULL, DAV1D_ERR(EINVAL));

    buf->ref = dav1d_ref_wrap(ptr, free_callback, cookie);
    if (!buf->ref) return DAV1D_ERR(ENOMEM);
    buf->data = ptr;
    buf->sz = buf->m.size = sz;
    dav1d_data_props_set_defaults(&buf->m);

    return 0;
}

int dav1d_data_wrap_user_data_internal(Dav1dData *const buf,
                                       const uint8_t *const user_data,
                                       void (*const free_callback)(const uint8_t *user_data,
                                                                   void *cookie),
                                       void *const cookie)
{
    validate_input_or_ret(buf != NULL, DAV1D_ERR(EINVAL));
    validate_input_or_ret(free_callback != NULL, DAV1D_ERR(EINVAL));

    buf->m.user_data.ref = dav1d_ref_wrap(user_data, free_callback, cookie);
    if (!buf->m.user_data.ref) return DAV1D_ERR(ENOMEM);
    buf->m.user_data.data = user_data;

    return 0;
}


void dav1d_data_ref(Dav1dData *const dst, const Dav1dData *const src) {
    validate_input(dst != NULL);
    validate_input(dst->data == NULL);
    validate_input(src != NULL);

    if (src->ref) {
        validate_input(src->data != NULL);
        dav1d_ref_inc(src->ref);
    }
    if (src->m.user_data.ref) dav1d_ref_inc(src->m.user_data.ref);
    *dst = *src;
}

void dav1d_data_props_copy(Dav1dDataProps *const dst,
                           const Dav1dDataProps *const src)
{
    assert(dst != NULL);
    assert(src != NULL);

    dav1d_ref_dec(&dst->user_data.ref);
    *dst = *src;
    if (dst->user_data.ref) dav1d_ref_inc(dst->user_data.ref);
}

void dav1d_data_props_set_defaults(Dav1dDataProps *const props) {
    assert(props != NULL);

    memset(props, 0, sizeof(*props));
    props->timestamp = INT64_MIN;
    props->offset = -1;
}

void dav1d_data_props_unref_internal(Dav1dDataProps *const props) {
    validate_input(props != NULL);

    struct Dav1dRef *user_data_ref = props->user_data.ref;
    dav1d_data_props_set_defaults(props);
    dav1d_ref_dec(&user_data_ref);
}

void dav1d_data_unref_internal(Dav1dData *const buf) {
    validate_input(buf != NULL);

    struct Dav1dRef *user_data_ref = buf->m.user_data.ref;
    if (buf->ref) {
        validate_input(buf->data != NULL);
        dav1d_ref_dec(&buf->ref);
    }
    memset(buf, 0, sizeof(*buf));
    dav1d_data_props_set_defaults(&buf->m);
    dav1d_ref_dec(&user_data_ref);
}
