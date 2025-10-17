/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
#ifndef __L2TP_UDP_H__
#define __L2TP_UDP_H__


int l2tp_udp_init(void);
int l2tp_udp_dispose(void);
int l2tp_udp_attach(socket_t *so, struct sockaddr *addr, int *thread, int nocksum, int delegated_process);
void l2tp_udp_detach(socket_t socket, int thread);
void l2tp_udp_retain(socket_t socket);
void l2tp_udp_socket_close(socket_t socket);
int l2tp_udp_setpeer(socket_t so, struct sockaddr *addr);
int l2tp_udp_output(socket_t so, int thread, mbuf_t m, struct sockaddr* to);
void l2tp_udp_input(socket_t so, void *arg, int waitflag);
void l2tp_udp_clear_INP_INADDR_ANY(socket_t so);

#endif
