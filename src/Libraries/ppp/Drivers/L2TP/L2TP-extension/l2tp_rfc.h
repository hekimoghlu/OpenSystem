/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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
#ifndef __L2TP_RFC_H__
#define __L2TP_RFC_H__

#define L2TP_MTU	1500

enum {
    L2TP_EVT_XMIT_OK = 1,
    L2TP_EVT_XMIT_FULL,
    L2TP_EVT_INPUTERROR,
    L2TP_EVT_RELIABLE_FAILED
};

enum {
    L2TP_CMD_VOID = 1,	 	// command codes to define
    L2TP_CMD_SETFLAGS,		// set flags
    L2TP_CMD_GETFLAGS,		// get flags
    L2TP_CMD_SETPEERADDR,	// set peer IP address
    L2TP_CMD_GETPEERADDR,	// get peer IP address
    L2TP_CMD_SETTUNNELID,	// set tunnel id
    L2TP_CMD_GETTUNNELID,	// get tunnel id
    L2TP_CMD_GETNEWTUNNELID,	// create, asign and return a new tunnel id
    L2TP_CMD_SETPEERTUNNELID,	// set peer tunnel id
    L2TP_CMD_SETSESSIONID,	// set session id
    L2TP_CMD_GETSESSIONID,	// get session id
    L2TP_CMD_SETPEERSESSIONID,	// set peer session id
    L2TP_CMD_SETWINDOW,		// set our receive window
    L2TP_CMD_SETPEERWINDOW,	// set peer receive window
    L2TP_CMD_SETTIMEOUT,	// set initial timeout value
    L2TP_CMD_SETTIMEOUTCAP,	// set timeout cap
    L2TP_CMD_SETMAXRETRIES,	// set max retries	
    L2TP_CMD_ACCEPT,		// accept connection request and xfer to new socket
    L2TP_CMD_SETOURADDR,	// set our IP address
    L2TP_CMD_GETOURADDR,	// get our IP address
    L2TP_CMD_SETBAUDRATE,	// set tunnel baud rate
    L2TP_CMD_GETBAUDRATE,	// get tunnel baud rate
    L2TP_CMD_SETRELIABILITY, // turn on/off the reliability layer
    L2TP_CMD_SETDELEGATEDPID // set the delegated process ID
};

typedef int (*l2tp_rfc_input_callback)(void *data, mbuf_t m, struct sockaddr *from, int more);
typedef void (*l2tp_rfc_event_callback)(void *data, u_int32_t evt, void *msg);

u_int16_t l2tp_rfc_init(void);
u_int16_t l2tp_rfc_dispose(void);
u_int16_t l2tp_rfc_new_client(void *host, void **data,
                         l2tp_rfc_input_callback input, 
                         l2tp_rfc_event_callback event);

void l2tp_rfc_free_client(void *data);
void l2tp_rfc_slowtimer(void);
u_int16_t l2tp_rfc_command(void *userdata, u_int32_t cmd, void *cmddata);
u_int16_t l2tp_rfc_output(void *data, mbuf_t m, struct sockaddr *to);

// callback from dlil layer
int l2tp_rfc_lower_input(socket_t so, mbuf_t m, struct sockaddr *from);

#endif
