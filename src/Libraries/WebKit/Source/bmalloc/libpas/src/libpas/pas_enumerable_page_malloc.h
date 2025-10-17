/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#ifndef PAS_ENUMERABLE_PAGE_MALLOC_H
#define PAS_ENUMERABLE_PAGE_MALLOC_H

#include "pas_aligned_allocation_result.h"
#include "pas_alignment.h"
#include "pas_enumerable_range_list.h"

PAS_BEGIN_EXTERN_C;

PAS_API extern pas_enumerable_range_list pas_enumerable_page_malloc_page_list;

/* It's assumed that whatever is returned from this is never deallocated, but may be decommitted. */
PAS_API pas_aligned_allocation_result
pas_enumerable_page_malloc_try_allocate_without_deallocating_padding(
    size_t size, pas_alignment alignment, bool may_contain_small_or_medium);

PAS_END_EXTERN_C;

#endif /* PAS_ENUMERABLE_PAGE_MALLOC_H */


