/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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
#ifndef PAS_LARGE_SHARING_POOL_EPOCH_UPDATE_MODE_H
#define PAS_LARGE_SHARING_POOL_EPOCH_UPDATE_MODE_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_large_sharing_pool_epoch_update_mode {
    pas_large_sharing_pool_forward_min_epoch,
    pas_large_sharing_pool_combined_use_epoch
};

typedef enum pas_large_sharing_pool_epoch_update_mode pas_large_sharing_pool_epoch_update_mode;

static inline const char*
pas_large_sharing_pool_epoch_update_mode_get_string(
    pas_large_sharing_pool_epoch_update_mode mode)
{
    switch (mode) {
    case pas_large_sharing_pool_forward_min_epoch:
        return "forward_min_epoch";
    case pas_large_sharing_pool_combined_use_epoch:
        return "combined_use_epoch";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_LARGE_SHARING_POOL_EPOCH_UPDATE_MODE_H */

