/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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
//  pgokext.c
//  pgokext
//
//  Created by Lawrence D'Anna on 12/15/16.
//
//

#include <mach/mach_types.h>

kern_return_t pgokext_start(kmod_info_t * ki, void *d);
kern_return_t pgokext_stop(kmod_info_t *ki, void *d);

kern_return_t
pgokext_start(kmod_info_t * ki, void *d)
{
	return KERN_SUCCESS;
}

kern_return_t
pgokext_stop(kmod_info_t *ki, void *d)
{
	return KERN_SUCCESS;
}
