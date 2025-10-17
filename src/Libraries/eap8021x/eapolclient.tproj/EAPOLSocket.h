/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
 * Modification History
 *
 * October 26, 2001	Dieter Siegmund (dieter@apple.com)
 * - created
 */

#ifndef _S_EAPOLSOCKET_H
#define _S_EAPOLSOCKET_H


#include "EAPOL.h"
#include "EAPOLControlTypes.h"
#include "wireless.h"
#include "ClientControlInterface.h"

typedef struct EAPOLSocket_s EAPOLSocket, * EAPOLSocketRef;

typedef struct {
    EAPOLPacket *		eapol_p;
    unsigned int		length;
} EAPOLSocketReceiveData, *EAPOLSocketReceiveDataRef;

typedef void (EAPOLSocketReceiveCallback)(void * arg1, void * arg2, 
					  EAPOLSocketReceiveDataRef data);
typedef EAPOLSocketReceiveCallback * EAPOLSocketReceiveCallbackRef;

void
EAPOLSocketSetDebug(boolean_t debug);

bool
EAPOLSocketIsLinkActive(EAPOLSocketRef sock);

int
EAPOLSocketMTU(EAPOLSocketRef sock);

const struct ether_addr *
EAPOLSocketGetAuthenticatorMACAddress(EAPOLSocketRef sock);

boolean_t
EAPOLSocketIsWireless(EAPOLSocketRef sock);

boolean_t
EAPOLSocketSetKey(EAPOLSocketRef sock, wirelessKeyType type,
		  int index, const uint8_t * key, int key_length);

boolean_t
EAPOLSocketSetWPAKey(EAPOLSocketRef sock, 
		     const uint8_t * msk, int msk_length);

void
EAPOLSocketClearPMKCache(EAPOLSocketRef sock);

boolean_t
EAPOLSocketHasPMK(EAPOLSocketRef sock);

CFStringRef
EAPOLSocketGetSSID(EAPOLSocketRef sock);

void
EAPOLSocketDisableReceive(EAPOLSocketRef eapol_socket);

void
EAPOLSocketEnableReceive(EAPOLSocketRef eapol_socket,
			 EAPOLSocketReceiveCallback * func,
			 void * arg1, void * arg2);

int
EAPOLSocketTransmit(EAPOLSocketRef sock,
		    EAPOLPacketType packet_type,
		    void * body, unsigned int body_length);

const char *
EAPOLSocketIfName(EAPOLSocketRef sock, uint32_t * length);

const char *
EAPOLSocketName(EAPOLSocketRef sock);

void
EAPOLSocketReportStatus(EAPOLSocketRef sock, CFDictionaryRef status_dict);

EAPOLControlMode
EAPOLSocketGetMode(EAPOLSocketRef sock);

void
EAPOLSocketStopClient(EAPOLSocketRef sock);

boolean_t
EAPOLSocketReassociate(EAPOLSocketRef sock);

int
get_plist_int(CFDictionaryRef plist, CFStringRef key, int def);

void
EAPOLSocketSourceUpdateWiFiLocalMACAddress(EAPOLSocketRef sock);

#endif /* _S_EAPOLSOCKET_H */

