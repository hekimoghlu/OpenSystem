/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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
#ifndef ISO_HEAP_INNARDS_H
#define ISO_HEAP_INNARDS_H

#include "pas_config.h"
#include "pas_allocator_counts.h"
#include "pas_dynamic_primitive_heap_map.h"
#include "pas_intrinsic_heap_support.h"
#include "pas_heap_ref.h"

#if PAS_ENABLE_ISO

PAS_BEGIN_EXTERN_C;

PAS_API extern pas_heap iso_common_primitive_heap;
PAS_API extern pas_intrinsic_heap_support iso_common_primitive_heap_support;
PAS_API extern pas_allocator_counts iso_allocator_counts;

PAS_API extern pas_dynamic_primitive_heap_map iso_primitive_dynamic_heap_map;
PAS_API extern pas_dynamic_primitive_heap_map iso_flex_dynamic_heap_map;

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_ISO */

#endif /* ISO_HEAP_INNARDS_h */

