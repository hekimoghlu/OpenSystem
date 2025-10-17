/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
#ifndef _ccid_ifd_handler_h_
#define _ccid_ifd_handler_h_

#define IOCTL_SMARTCARD_VENDOR_IFD_EXCHANGE	SCARD_CTL_CODE(1)

#define CLASS2_IOCTL_MAGIC 0x330000
#define IOCTL_FEATURE_VERIFY_PIN_DIRECT \
	SCARD_CTL_CODE(FEATURE_VERIFY_PIN_DIRECT + CLASS2_IOCTL_MAGIC)
#define IOCTL_FEATURE_MODIFY_PIN_DIRECT \
	SCARD_CTL_CODE(FEATURE_MODIFY_PIN_DIRECT + CLASS2_IOCTL_MAGIC)
#define IOCTL_FEATURE_MCT_READER_DIRECT \
	SCARD_CTL_CODE(FEATURE_MCT_READER_DIRECT + CLASS2_IOCTL_MAGIC)
#define IOCTL_FEATURE_IFD_PIN_PROPERTIES \
	SCARD_CTL_CODE(FEATURE_IFD_PIN_PROPERTIES + CLASS2_IOCTL_MAGIC)
#define IOCTL_FEATURE_GET_TLV_PROPERTIES \
	SCARD_CTL_CODE(FEATURE_GET_TLV_PROPERTIES + CLASS2_IOCTL_MAGIC)

#define DRIVER_OPTION_CCID_EXCHANGE_AUTHORIZED 1
#define DRIVER_OPTION_GEMPC_TWIN_KEY_APDU 2
#define DRIVER_OPTION_USE_BOGUS_FIRMWARE 4
#define DRIVER_OPTION_DISABLE_PIN_RETRIES (1 << 6)

extern int DriverOptions;

/*
 * Maximum number of CCID readers supported simultaneously
 *
 * The maximum number of readers is also limited in pcsc-lite (16 by default)
 * see the definition of PCSCLITE_MAX_READERS_CONTEXTS in src/PCSC/pcsclite.h
 */
#define CCID_DRIVER_MAX_READERS 16

/*
 * CCID driver specific functions
 */
CcidDesc *get_ccid_slot(unsigned int reader_index);

#endif

