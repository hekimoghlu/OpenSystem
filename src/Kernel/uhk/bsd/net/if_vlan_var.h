/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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
 * Copyright 1998 Massachusetts Institute of Technology
 *
 * Permission to use, copy, modify, and distribute this software and
 * its documentation for any purpose and without fee is hereby
 * granted, provided that both the above copyright notice and this
 * permission notice appear in all copies, that both the above
 * copyright notice and this permission notice appear in all
 * supporting documentation, and that the name of M.I.T. not be used
 * in advertising or publicity pertaining to distribution of the
 * software without specific, written prior permission.  M.I.T. makes
 * no representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied
 * warranty.
 *
 * THIS SOFTWARE IS PROVIDED BY M.I.T. ``AS IS''.  M.I.T. DISCLAIMS
 * ALL EXPRESS OR IMPLIED WARRANTIES WITH REGARD TO THIS SOFTWARE,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT
 * SHALL M.I.T. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * $FreeBSD: src/sys/net/if_vlan_var.h,v 1.16 2003/07/08 21:54:20 wpaul Exp $
 */

#ifndef _NET_IF_VLAN_VAR_H_
#define _NET_IF_VLAN_VAR_H_     1

#include <net/ethernet.h>
#include <net/if_var.h>
#include <sys/types.h>

#define ETHER_VLAN_ENCAP_LEN    4       /* len of 802.1Q VLAN encapsulation */

#ifdef KERNEL_PRIVATE
/* VLAN encapsulation header */
struct ether_vlan_encap_header {
	u_int16_t evle_tag;
	u_int16_t evle_proto;
};
#endif /* KERNEL_PRIVATE */

struct  ether_vlan_header {
	u_char  evl_dhost[ETHER_ADDR_LEN];
	u_char  evl_shost[ETHER_ADDR_LEN];
	u_int16_t evl_encap_proto;
	u_int16_t evl_tag;
	u_int16_t evl_proto;
};

#define EVL_VLID_MASK   0x0FFF
#define EVL_VLANOFTAG(tag) ((tag) & EVL_VLID_MASK)
#define EVL_PRIOFTAG(tag) (((tag) >> 13) & 7)

#if 0
/* sysctl(3) tags, for compatibility purposes */
#define VLANCTL_PROTO   1
#define VLANCTL_MAX     2
#endif

/*
 * Configuration structure for SIOCSETVLAN and SIOCGETVLAN ioctls.
 */
struct  vlanreq {
	char    vlr_parent[IFNAMSIZ];
	u_short vlr_tag;
};

#ifdef KERNEL_PRIVATE
int vlan_family_init(void);
#endif /* KERNEL_PRIVATE */
#endif /* _NET_IF_VLAN_VAR_H_ */
