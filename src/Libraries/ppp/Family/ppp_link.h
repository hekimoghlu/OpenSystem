/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#ifndef _PPP_LINK_H_
#define _PPP_LINK_H_


int ppp_link_init(void);
int ppp_link_dispose(void);
int ppp_link_control(struct ppp_link *link, u_long cmd, void *data);
int ppp_link_attachclient(u_short index, void *host, struct ppp_link **link);
int ppp_link_detachclient(struct ppp_link *link, void *host);
int ppp_link_send(struct ppp_link *link, mbuf_t m);


#endif /* _PPP_LINK_H_ */
