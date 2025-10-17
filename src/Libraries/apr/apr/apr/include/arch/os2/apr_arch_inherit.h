/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
#ifndef INHERIT_H
#define INHERIT_H

#include "apr_inherit.h"

#define APR_INHERIT (1 << 24)    /* Must not conflict with other bits */

#define APR_IMPLEMENT_INHERIT_SET(name, flag, pool, cleanup)        \
APR_DECLARE(apr_status_t) apr_##name##_inherit_set(apr_##name##_t *the##name) \
{                                                                   \
    int rv;                                                         \
    ULONG state;                                                    \
    if (((rv = DosQueryFHState(attr->parent_err->filedes, &state))  \
            != 0) ||                                                \
        ((rv = DosSetFHState(attr->parent_err->filedes,             \
                            state & ~OPEN_FLAGS_NOINHERIT)) != 0))  \
        return APR_FROM_OS_ERROR(rv);                               \
    return APR_SUCCESS;                                             \
}

#define APR_IMPLEMENT_INHERIT_UNSET(name, flag, pool, cleanup)      \
APR_DECLARE(apr_status_t) apr_##name##_inherit_unset(apr_##name##_t *the##name)\
{                                                                   \
    int rv;                                                         \
    ULONG state;                                                    \
    if (((rv = DosQueryFHState(attr->parent_err->filedes, &state))  \
            != 0) ||                                                \
        ((rv = DosSetFHState(attr->parent_err->filedes,             \
                            state | OPEN_FLAGS_NOINHERIT)) != 0))   \
        return APR_FROM_OS_ERROR(rv);                               \
    return APR_SUCCESS;                                             \
}

#endif	/* ! INHERIT_H */
