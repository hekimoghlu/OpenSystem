/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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
#ifndef PAS_PAYLOAD_RESERVATION_PAGE_LIST_H
#define PAS_PAYLOAD_RESERVATION_PAGE_LIST_H

#include "pas_enumerable_range_list.h"

PAS_BEGIN_EXTERN_C;

PAS_API extern pas_enumerable_range_list pas_payload_reservation_page_list;

/* In cases where we would do a reservation for object payloads that isn't tracked by the large sharing
   pool or segregated/bitfit, we add it here. This is used to exclude that memory from being tracked as
   meta during enumeration. */
PAS_API void pas_payload_reservation_page_list_append(pas_range range);

PAS_END_EXTERN_C;

#endif /* PAS_PAYLOAD_RESERVATION_PAGE_LIST_H */

