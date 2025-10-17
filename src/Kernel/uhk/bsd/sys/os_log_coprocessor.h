/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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
 * @OSF_COPYRIGHT@
 */
/*
 */

#ifndef _OS_LOG_COPROCESSOR_H_
#define _OS_LOG_COPROCESSOR_H_

#include <stdbool.h>
#include <os/log.h>
#include <sys/types.h>

/*
 * Userspace coprocessor logging
 *
 * Description: sends a syscall ending up at the kernel `os_log_coprocessor` function
 *
 */
int os_log_coprocessor_as_kernel(void *buff, uint64_t buff_len, os_log_type_t type, const char *uuid, uint64_t timestamp, uint32_t offset, bool stream_log);

/*
 * Userspace coprocessor logging registration
 *
 * Description: sends a syscall ending up at the kernel `os_log_coprocessor_register_with_type` function
 *
 */
int os_log_coprocessor_register_as_kernel(const char *uuid, const char *file_path, size_t file_path_len);

#endif /* _OS_LOG_COPROCESSOR_H_ */
