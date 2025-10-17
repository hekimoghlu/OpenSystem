/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
#include <sys/os_log_coprocessor.h>
#include <strings.h>

extern int
__oslog_coproc(void *buff, uint64_t buff_len, uint32_t type, const char *uuid, uint64_t timestamp, uint32_t offset, uint32_t stream_log);

extern int
__oslog_coproc_reg(const char *uuid, const char *file_path, size_t file_path_len);

int
os_log_coprocessor_as_kernel(void *buff, uint64_t buff_len, os_log_type_t type, const char *uuid, uint64_t timestamp, uint32_t offset, bool stream_log)
{
	return __oslog_coproc(buff, buff_len, type, uuid, timestamp, offset, stream_log);
}

int
os_log_coprocessor_register_as_kernel(const char *uuid, const char *file_path, size_t file_path_len)
{
	return __oslog_coproc_reg(uuid, file_path, file_path_len);
}
