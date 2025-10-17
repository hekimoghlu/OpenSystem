/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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
#include "apr_buckets.h"

APU_DECLARE_NONSTD(apr_status_t) apr_bucket_setaside_noop(apr_bucket *data,
                                                          apr_pool_t *pool)
{
    return APR_SUCCESS;
}

APU_DECLARE_NONSTD(apr_status_t) apr_bucket_setaside_notimpl(apr_bucket *data,
                                                             apr_pool_t *pool)
{
    return APR_ENOTIMPL;
}

APU_DECLARE_NONSTD(apr_status_t) apr_bucket_split_notimpl(apr_bucket *data,
                                                          apr_size_t point)
{
    return APR_ENOTIMPL;
}

APU_DECLARE_NONSTD(apr_status_t) apr_bucket_copy_notimpl(apr_bucket *e,
                                                         apr_bucket **c)
{
    return APR_ENOTIMPL;
}

APU_DECLARE_NONSTD(void) apr_bucket_destroy_noop(void *data)
{
    return;
}
