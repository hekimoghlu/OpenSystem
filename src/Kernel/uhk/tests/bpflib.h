/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#ifndef _S_BPFLIB_H
#define _S_BPFLIB_H

#include <sys/types.h>

int bpf_get_blen(int fd, int * blen);
int bpf_set_blen(int fd, int blen);
int bpf_new(void);
int bpf_dispose(int fd);
int bpf_setif(int fd, const char * en_name);
int bpf_set_immediate(int fd, u_int value);
int bpf_filter_receive_none(int fd);
int bpf_arp_filter(int fd, int type_offset, int type, u_int packet_size);
int bpf_set_timeout(int fd, struct timeval * tv_p);
int bpf_set_header_complete(int fd, u_int header_complete);
int bpf_set_see_sent(int fd, u_int see_send);
int bpf_set_traffic_class(int fd, int tc);
int bpf_set_direction(int fd, u_int direction);
int bpf_get_direction(int fd, u_int *direction);
int bpf_set_write_size_max(int fd, u_int write_size_max);
int bpf_get_write_size_max(int fd, u_int *write_size_max);
int bpf_set_batch_write(int fd, u_int batch_write);
int bpf_get_batch_write(int fd, u_int *batch_write);

#endif /* _S_BPFLIB_H */
