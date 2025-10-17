/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_segregated_page_config.h"

#include "pas_config.h"
#include "pas_page_malloc.h"
#include "pas_segregated_page.h"

bool pas_segregated_page_config_do_validate = false;

bool pas_small_segregated_page_config_variant_is_enabled_override =
    PAS_USE_SMALL_SEGREGATED_OVERRIDE;
bool pas_medium_segregated_page_config_variant_is_enabled_override =
    PAS_USE_MEDIUM_SEGREGATED_OVERRIDE;

void pas_segregated_page_config_validate(const pas_segregated_page_config* config)
{
    if (!pas_segregated_page_config_do_validate)
        return;

    PAS_ASSERT(config->exclusive_payload_size <= config->base.page_size);
    PAS_ASSERT(config->shared_payload_size <= config->base.page_size);
    PAS_ASSERT(pas_segregated_page_config_min_align(*config) < config->base.max_object_size);
    PAS_ASSERT(config->exclusive_payload_offset < config->base.page_size);
    PAS_ASSERT(config->shared_payload_offset < config->base.page_size);
    PAS_ASSERT(config->base.max_object_size <= config->exclusive_payload_size);
    PAS_ASSERT(config->base.max_object_size <= config->shared_payload_size);
    PAS_ASSERT(config->num_alloc_bits >=
               (pas_segregated_page_config_payload_end_offset_for_role(
                   *config, pas_segregated_page_shared_role) >> config->base.min_align_shift));
    PAS_ASSERT(pas_segregated_page_config_payload_end_offset_for_role(
                   *config, pas_segregated_page_exclusive_role) <= config->base.page_size);
    PAS_ASSERT(!(config->base.page_size % config->base.granule_size));
    PAS_ASSERT(config->base.page_size >= config->base.granule_size);
    PAS_ASSERT(!(config->base.granule_size % pas_page_malloc_alignment()));
    PAS_ASSERT(config->base.granule_size >= pas_page_malloc_alignment());
    if (config->base.page_size > config->base.granule_size)
        PAS_ASSERT(((config->base.granule_size >> config->base.min_align_shift) + 1)
                   < PAS_PAGE_GRANULE_DECOMMITTED);
}

#endif /* LIBPAS_ENABLED */
