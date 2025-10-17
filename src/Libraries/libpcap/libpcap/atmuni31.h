/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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
/* Based on UNI3.1 standard by ATM Forum */

/* ATM traffic types based on VPI=0 and (the following VCI */
#define VCI_PPC			0x05	/* Point-to-point signal msg */
#define VCI_BCC			0x02	/* Broadcast signal msg */
#define VCI_OAMF4SC		0x03	/* Segment OAM F4 flow cell */
#define VCI_OAMF4EC		0x04	/* End-to-end OAM F4 flow cell */
#define VCI_METAC		0x01	/* Meta signal msg */
#define VCI_ILMIC		0x10	/* ILMI msg */

/* Q.2931 signalling messages */
#define CALL_PROCEED		0x02	/* call proceeding */
#define CONNECT			0x07	/* connect */
#define CONNECT_ACK		0x0f	/* connect_ack */
#define SETUP			0x05	/* setup */
#define RELEASE			0x4d	/* release */
#define RELEASE_DONE		0x5a	/* release_done */
#define RESTART			0x46	/* restart */
#define RESTART_ACK		0x4e	/* restart ack */
#define STATUS			0x7d	/* status */
#define STATUS_ENQ		0x75	/* status ack */
#define ADD_PARTY		0x80	/* add party */
#define ADD_PARTY_ACK		0x81	/* add party ack */
#define ADD_PARTY_REJ		0x82	/* add party rej */
#define DROP_PARTY		0x83	/* drop party */
#define DROP_PARTY_ACK		0x84	/* drop party ack */

/* Information Element Parameters in the signalling messages */
#define CAUSE			0x08	/* cause */
#define ENDPT_REF		0x54	/* endpoint reference */
#define AAL_PARA		0x58	/* ATM adaptation layer parameters */
#define TRAFF_DESCRIP		0x59	/* atm traffic descriptors */
#define CONNECT_ID		0x5a	/* connection identifier */
#define QOS_PARA		0x5c	/* quality of service parameters */
#define B_HIGHER		0x5d	/* broadband higher layer information */
#define B_BEARER		0x5e	/* broadband bearer capability */
#define B_LOWER			0x5f	/* broadband lower information */
#define CALLING_PARTY		0x6c	/* calling party number */
#define CALLED_PARTY		0x70	/* called party nmber */

#define Q2931			0x09

/* Q.2931 signalling general messages format */
#define PROTO_POS       0	/* offset of protocol discriminator */
#define CALL_REF_POS    2	/* offset of call reference value */
#define MSG_TYPE_POS    5	/* offset of message type */
#define MSG_LEN_POS     7	/* offset of message length */
#define IE_BEGIN_POS    9	/* offset of first information element */

/* format of signalling messages */
#define TYPE_POS	0
#define LEN_POS		2
#define FIELD_BEGIN_POS 4
