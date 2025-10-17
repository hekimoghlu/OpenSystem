/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
#ifndef _VM_VM_COMPRESSOR_BACKING_STORE_XNU_H_
#define _VM_VM_COMPRESSOR_BACKING_STORE_XNU_H_

#ifdef XNU_KERNEL_PRIVATE

uint64_t vm_swap_get_total_space(void);
uint64_t vm_swap_get_free_space(void);

#if CONFIG_FREEZE
boolean_t vm_swap_max_budget(uint64_t *);
#endif /* CONFIG_FREEZE */

#endif /* XNU_KERNEL_PRIVATE */
#endif /* _VM_VM_COMPRESSOR_BACKING_STORE_XNU_H_ */
