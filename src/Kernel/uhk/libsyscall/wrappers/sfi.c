/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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
#include <sys/errno.h>
#include <sys/sfi.h>
#include <sys/types.h>
#include <unistd.h>

int
system_set_sfi_window(uint64_t sfi_window_usec)
{
	return __sfi_ctl(SFI_CTL_OPERATION_SFI_SET_WINDOW, 0, sfi_window_usec, NULL);
}

int
system_get_sfi_window(uint64_t *sfi_window_usec)
{
	return __sfi_ctl(SFI_CTL_OPERATION_SFI_GET_WINDOW, 0, 0, sfi_window_usec);
}

int
sfi_set_class_offtime(sfi_class_id_t class_id, uint64_t offtime_usec)
{
	return __sfi_ctl(SFI_CTL_OPERATION_SET_CLASS_OFFTIME, class_id, offtime_usec, NULL);
}

int
sfi_get_class_offtime(sfi_class_id_t class_id, uint64_t *offtime_usec)
{
	return __sfi_ctl(SFI_CTL_OPERATION_GET_CLASS_OFFTIME, class_id, 0, offtime_usec);
}

int
sfi_process_set_flags(pid_t pid, uint32_t flags)
{
	return __sfi_pidctl(SFI_PIDCTL_OPERATION_PID_SET_FLAGS, pid, flags, NULL);
}

int
sfi_process_get_flags(pid_t pid, uint32_t *flags)
{
	return __sfi_pidctl(SFI_PIDCTL_OPERATION_PID_GET_FLAGS, pid, 0, flags);
}
