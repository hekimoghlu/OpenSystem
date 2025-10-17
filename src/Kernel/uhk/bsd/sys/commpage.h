/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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
#ifndef _COMMPAGE_H
#define _COMMPAGE_H

#ifdef  PRIVATE

#include <stdint.h>

#define _COMM_PAGE32_SIGNATURE_STRING           "commpage 32-bit"
#define _COMM_PAGE64_SIGNATURE_STRING           "commpage 64-bit"

typedef volatile struct commpage_timeofday_data {
	uint64_t        TimeStamp_tick;
	uint64_t        TimeStamp_sec;
	uint64_t        TimeStamp_frac;
	uint64_t        Ticks_scale;
	uint64_t        Ticks_per_sec;
} new_commpage_timeofday_data_t;

/*!
 * @macro COMM_PAGE_SLOT_TYPE
 *
 * @brief
 * Macro that expands to the proper type for a pointer to a commpage slot,
 * to be used in a local variable declaration.
 *
 * @description
 * Usage is something like:
 * <code>
 *     COMM_PAGE_SLOT_TYPE(uint64_t) slot = COMM_PAGE_SLOT(uint64_t, FOO);
 * </code>
 *
 * @param type   The scalar base type for the slot.
 */
#if __has_feature(address_sanitizer)
#define COMM_PAGE_SLOT_TYPE(type_t)     type_t __attribute__((address_space(1))) volatile *
#else
#define COMM_PAGE_SLOT_TYPE(type_t)     type_t volatile *
#endif

/*!
 * @macro COMM_PAGE_SLOT
 *
 * @brief
 * Macro that expands to the properly typed address for a commpage slot.
 *
 * @param type   The scalar base type for the slot.
 * @param name   The slot name, without its @c _COMM_PAGE_ prefix.
 */
#define COMM_PAGE_SLOT(type_t, name)    ((COMM_PAGE_SLOT_TYPE(type_t))_COMM_PAGE_##name)

/*!
 * @macro COMM_PAGE_READ
 *
 * @brief
 * Performs a single read from the commpage in a way that doesn't trip
 * address sanitizers.
 *
 * @description
 * Typical use looks like this:
 * <code>
 *     uint64_t foo_value = COMM_PAGE_READ(uint64_t, FOO);
 * </code>
 *
 * @param type   The scalar base type for the slot.
 * @param name   The slot name, without its @c _COMM_PAGE_ prefix.
 */
#define COMM_PAGE_READ(type_t, slot)    (*(COMM_PAGE_SLOT(type_t, slot)))

#endif

#endif
