/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#include <sys/csr.h>
#include <sys/types.h>

/* Syscall entry point */
int __csrctl(csr_op_t op, void *buffer, size_t size);

int
csr_check(csr_config_t mask)
{
	return __csrctl(CSR_SYSCALL_CHECK, &mask, sizeof(csr_config_t));
}

int
csr_get_active_config(csr_config_t *config)
{
	return __csrctl(CSR_SYSCALL_GET_ACTIVE_CONFIG, config, sizeof(csr_config_t));
}
