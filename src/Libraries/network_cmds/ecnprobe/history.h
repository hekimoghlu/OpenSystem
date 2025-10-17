/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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
#define SENT 1
#define RCVD 2
#define MAXHSZ 10000

struct History { /* store history of each packet as it is seen */

  int type ;        /* sent or received */
  double timestamp; /* when */
  uint32 seqno;
  uint32 nextbyte;  /* seqno + dlen */
  uint32 ackno;
  int hlen;
  int ecn_echo;
  int cwr;
  int urg;
  int ack;
  int psh;
  int rst;
  int syn;
  int fin;
  int ip_optlen;   /* added to support IP options */
  uint8 *ip_opt;   /* added to support IP options */
  int optlen;
  uint8 *opt;
  uint8 *data;
  int dlen;

  int pkt_num;

};

void StorePacket (struct IPPacket *p); 
int reordered (struct IPPacket *p);
