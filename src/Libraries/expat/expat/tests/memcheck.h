/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
#ifdef __cplusplus
extern "C" {
#endif

#ifndef XML_MEMCHECK_H
#  define XML_MEMCHECK_H 1

/* Allocation declarations */

void *tracking_malloc(size_t size);
void tracking_free(void *ptr);
void *tracking_realloc(void *ptr, size_t size);

/* End-of-test check to see if unfreed allocations remain. Returns
 * TRUE (1) if there is nothing, otherwise prints a report of the
 * remaining allocations and returns FALSE (0).
 */
int tracking_report(void);

#endif /* XML_MEMCHECK_H */

#ifdef __cplusplus
}
#endif
