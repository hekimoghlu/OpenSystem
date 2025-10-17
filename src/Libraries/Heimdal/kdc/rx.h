/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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
/* $Id$ */

#ifndef __RX_H__
#define __RX_H__

/* header of a RPC packet */

enum rx_header_type {
     HT_DATA = 1,
     HT_ACK = 2,
     HT_BUSY = 3,
     HT_ABORT = 4,
     HT_ACKALL = 5,
     HT_CHAL = 6,
     HT_RESP = 7,
     HT_DEBUG = 8
};

/* For flags in header */

enum rx_header_flag {
     HF_CLIENT_INITIATED = 1,
     HF_REQ_ACK = 2,
     HF_LAST = 4,
     HF_MORE = 8
};

struct rx_header {
     uint32_t epoch;
     uint32_t connid;		/* And channel ID */
     uint32_t callid;
     uint32_t seqno;
     uint32_t serialno;
     u_char type;
     u_char flags;
     u_char status;
     u_char secindex;
     uint16_t reserved;	/* ??? verifier? */
     uint16_t serviceid;
/* This should be the other way around according to everything but */
/* tcpdump */
};

#define RX_HEADER_SIZE 28

#endif /* __RX_H__ */
