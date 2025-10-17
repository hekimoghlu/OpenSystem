/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 20, 2023.
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
#define max( a, b )   ( ( ( a ) > ( b ) ) ? ( a ) : ( b ) )

#include <pcsclite.h>

#include "openct/proto-t1.h"

typedef struct CCID_DESC
{
	/*
	 * ATR
	 */
	int nATRLength;
	unsigned char pcATRBuffer[MAX_ATR_SIZE];

	/*
	 * Card state
	 */
	unsigned char bPowerFlags;

	/*
	 * T=1 Protocol context
	 */
	t1_state_t t1;

	/* reader name passed to IFDHCreateChannelByName() */
	char *readerName;
} CcidDesc;

typedef enum {
	STATUS_NO_SUCH_DEVICE        = 0xF9,
	STATUS_SUCCESS               = 0xFA,
	STATUS_UNSUCCESSFUL          = 0xFB,
	STATUS_COMM_ERROR            = 0xFC,
	STATUS_DEVICE_PROTOCOL_ERROR = 0xFD,
	STATUS_COMM_NAK              = 0xFE,
	STATUS_SECONDARY_SLOT        = 0xFF
} status_t;

/* Powerflag (used to detect quick insertion removals unnoticed by the
 * resource manager) */
/* Initial value */
#define POWERFLAGS_RAZ 0x00
/* Flag set when a power up has been requested */
#define MASK_POWERFLAGS_PUP 0x01
/* Flag set when a power down is requested */
#define MASK_POWERFLAGS_PDWN 0x02

/* Communication buffer size (max=adpu+Lc+data+Le)
 * we use a 64kB for extended APDU on APDU mode readers */
#define CMD_BUF_SIZE (4 +3 +64*1024 +3)

/* Protocols */
#define T_0 0
#define T_1 1

/* Default communication read timeout in milliseconds */
#define DEFAULT_COM_READ_TIMEOUT (3*1000)

/* DWORD type formating */
#ifdef __APPLE__
/* Apple defines DWORD as uint32_t */
#define DWORD_X "%X"
#define DWORD_D "%d"
#else
/* pcsc-lite defines DWORD as unsigned long */
#define DWORD_X "%lX"
#define DWORD_D "%ld"
#endif

/*
 * communication ports abstraction
 */
#ifdef TWIN_SERIAL

#define OpenPortByName OpenSerialByName
#define OpenPort OpenSerial
#define ClosePort CloseSerial
#define ReadPort ReadSerial
#define WritePort WriteSerial
#define DisconnectPort DisconnectSerial
#include "ccid_serial.h"

#else

#define OpenPortByName OpenUSBByName
#define OpenPort OpenUSB
#define ClosePort CloseUSB
#define ReadPort ReadUSB
#define WritePort WriteUSB
#define DisconnectPort DisconnectUSB
#include "ccid_usb.h"

#endif

