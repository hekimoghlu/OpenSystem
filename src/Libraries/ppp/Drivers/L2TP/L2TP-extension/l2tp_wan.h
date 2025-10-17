/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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
#ifndef __l2tp_WAN_H__
#define __l2tp_WAN_H__

int l2tp_wan_init(void);
int l2tp_wan_dispose(void);
int l2tp_wan_attach(void *rfc, struct ppp_link **link);
void l2tp_wan_detach(struct ppp_link *link);
int l2tp_wan_input(struct ppp_link *link, mbuf_t m);
void l2tp_wan_xmit_full(struct ppp_link *link);
void l2tp_wan_xmit_ok(struct ppp_link *link);
void l2tp_wan_input_error(struct ppp_link *);


#endif
