/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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
#ifdef KERNEL_PRIVATE
#ifndef _ARM_CPU_AFFINITY_H_
#define _ARM_CPU_AFFINITY_H_

static inline int
ml_get_max_affinity_sets(void)
{
	return 0;
}

static inline processor_set_t
ml_affinity_to_pset(__unused int affinity_num)
{
	return PROCESSOR_SET_NULL;
}

#endif /* _ARM_CPU_AFFINITY_H_ */
#endif /* KERNEL_PRIVATE */
