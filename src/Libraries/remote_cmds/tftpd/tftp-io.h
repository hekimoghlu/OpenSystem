/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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
#define RP_NONE		0
#define	RP_RECVFROM	-1
#define	RP_TOOSMALL	-2
#define RP_ERROR	-3
#define RP_WRONGSOURCE	-4
#define	RP_TIMEOUT	-5
#define	RP_TOOBIG	-6

const char *errtomsg(int);
void	send_error(int peer, int);
int	send_wrq(int peer, char *, char *);
int	send_rrq(int peer, char *, char *);
int	send_oack(int peer);
int	send_ack(int peer, unsigned short);
int	send_data(int peer, uint16_t, char *, int);
int	receive_packet(int peer, char *, int, struct sockaddr_storage *, int);

extern struct sockaddr_storage peer_sock;
extern struct sockaddr_storage me_sock;
