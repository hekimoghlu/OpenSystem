/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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
#ifndef __PPP_IP_H__
#define __PPP_IP_H__


int ppp_ip_init(int init_arg);
int ppp_ip_dispose(int term_arg);

errno_t ppp_ip_attach(ifnet_t ifp, protocol_family_t protocol_family);
void ppp_ip_detach(ifnet_t ifp, protocol_family_t protocol_family);

int ppp_ip_af_src_out(ifnet_t ifp, char *pkt);
int ppp_ip_af_src_in(ifnet_t ifp, char *pkt);

int ppp_ip_bootp_server_in(ifnet_t ifp, char *pkt);
int ppp_ip_bootp_client_in(ifnet_t ifp, char *pkt);

#endif
