/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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
/*
 * Copyright 1995 NeXT Computer, Inc. All rights reserved.
 */
#ifndef _BSD_MACHINE_EXEC_H_
#define _BSD_MACHINE_EXEC_H_

#include <sys/param.h>
#include <stdbool.h>

struct exec_info {
	char    path[MAXPATHLEN];
	int     ac;
	int     ec;
	char    **av;
	char    **ev;
};

int grade_binary(cpu_type_t, cpu_subtype_t, cpu_subtype_t, bool allow_simulator_binary);
boolean_t binary_match(cpu_type_t mask_bits, cpu_type_t req_cpu,
    cpu_subtype_t req_subcpu, cpu_type_t test_cpu,
    cpu_subtype_t test_subcpu);

#endif /* _BSD_MACHINE_EXEC_H_ */
