/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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

#include "pas_heap_summary.h"

#include "pas_stream.h"

void pas_heap_summary_validate(pas_heap_summary* summary)
{
    PAS_ASSERT(summary->free + summary->allocated <= summary->committed + summary->decommitted);
    PAS_ASSERT((summary->allocated +
                summary->meta_ineligible_for_decommit +
                summary->meta_eligible_for_decommit)
               == summary->committed);
    PAS_ASSERT((summary->free_ineligible_for_decommit +
                summary->free_eligible_for_decommit +
                summary->free_decommitted)
               == summary->free);
    PAS_ASSERT(summary->free_ineligible_for_decommit + summary->free_eligible_for_decommit
               <= summary->committed);
    PAS_ASSERT(summary->free_decommitted <= summary->decommitted);
    PAS_ASSERT(summary->cached <= summary->committed);
}

void pas_heap_summary_dump(pas_heap_summary summary, pas_stream* stream)
{
    pas_stream_printf(
        stream,
        "%.0lf%% Alloc: %zu/%zu (CO)/%zu (CT)/%zu (R); Frag: %zu (%.0lf%%)",
        pas_heap_summary_total(summary)
        ? 100. * (summary.allocated + summary.meta) / pas_heap_summary_total(summary)
        : 0.,
        summary.allocated,
        pas_heap_summary_committed_objects(summary),
        summary.committed,
        pas_heap_summary_total(summary),
        pas_heap_summary_fragmentation(summary),
        summary.committed
        ? 100. * pas_heap_summary_fragmentation(summary) / summary.committed
        : 0.);

    if (summary.cached)
        pas_stream_printf(stream, "; Cached: %zu", summary.cached);
}

#endif /* LIBPAS_ENABLED */
