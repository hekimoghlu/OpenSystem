/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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
/*	$NetBSD: lockd.h,v 1.2 2000/06/07 14:34:40 bouyer Exp $	*/
/*	$FreeBSD: src/usr.sbin/rpc.lockd/lockd.h,v 1.5 2001/11/29 17:36:45 alfred Exp $ */

/*
 * Copyright (c) 1995
 *	A.R. Gordon (andrew.gordon@net-tel.co.uk).  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed for the FreeBSD project
 * 4. Neither the name of the author nor the names of any co-contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY ANDREW GORDON AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */

/*
 * structure holding lockd config values
 */
struct nfs_conf_lockd {
	int grace_period;
	int host_monitor_cache_timeout;
	int port;
	int send_using_tcp;
	int send_using_mnt_transport;
	int shutdown_delay_client;
	int shutdown_delay_server;
	int tcp;
	int udp;
	int verbose;
};

extern struct nfs_conf_lockd l_config;
extern int grace_expired;
extern int nsm_state;
extern time_t currsec;

#define _PATH_LOCKD_PID                 "/var/run/lockd.pid"
#define _PATH_STATD_PID                 "/var/run/statd.pid"
#define _PATH_NFS_CONF                  "/etc/nfs.conf"
#define _STATD_SERVICE_LABEL            "com.apple.statd"
#define _PATH_STATD                     "/usr/sbin/rpc.statd"

void client_mach_request(void);
int expire_lock_hosts(void);
void handle_sig_cleanup(int);
int statd_start(void);
int statd_stop(void);

/* XXX these should be in some system headers */
typedef u_int32_t rpcvers_t;
