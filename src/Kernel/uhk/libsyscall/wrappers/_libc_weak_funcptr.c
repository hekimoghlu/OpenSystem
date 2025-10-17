/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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
//
//  _libc_weak_funcptr.c
//  Libsyscall_static
//
//  Created by Ian Fang on 11/30/21.
//
//  dyld needs the following definitions to link against Libsyscall_static.
//  When building Libsyscall_dynamic, the weak symbols below will get overridden
//  by actual implementation.
//

#include "_libkernel_init.h"

__attribute__((weak, visibility("hidden")))
void *
malloc(__unused size_t size)
{
	return NULL;
}

__attribute__((weak, visibility("hidden")))
mach_msg_size_t
voucher_mach_msg_fill_aux(__unused mach_msg_aux_header_t *aux_hdr,
    __unused mach_msg_size_t sz)
{
	return 0;
}

__attribute__((weak, visibility("hidden")))
boolean_t
voucher_mach_msg_fill_aux_supported(void)
{
	return FALSE;
}
