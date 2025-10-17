/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#define SIZE_GET_SLOT_STATUS 10
#define STATUS_OFFSET 7
#define ERROR_OFFSET 8
#define CHAIN_PARAMETER_OFFSET 9
#define CCID_RESPONSE_HEADER_SIZE 10

RESPONSECODE CmdPowerOn(unsigned int reader_index, unsigned int * nlength,
	/*@out@*/ unsigned char buffer[], int voltage);

RESPONSECODE SecurePINVerify(unsigned int reader_index,
	unsigned char TxBuffer[], unsigned int TxLength,
	unsigned char RxBuffer[], unsigned int *RxLength);

RESPONSECODE SecurePINModify(unsigned int reader_index,
	unsigned char TxBuffer[], unsigned int TxLength,
	unsigned char RxBuffer[], unsigned int *RxLength);

RESPONSECODE CmdEscape(unsigned int reader_index,
	const unsigned char TxBuffer[], unsigned int TxLength,
	unsigned char RxBuffer[], unsigned int *RxLength, unsigned int timeout);

RESPONSECODE CmdEscapeCheck(unsigned int reader_index,
	const unsigned char TxBuffer[], unsigned int TxLength,
	unsigned char RxBuffer[], unsigned int *RxLength, unsigned int timeout,
	bool mayfail);

RESPONSECODE CmdPowerOff(unsigned int reader_index);

RESPONSECODE CmdGetSlotStatus(unsigned int reader_index,
	/*@out@*/ unsigned char buffer[]);

RESPONSECODE CmdXfrBlock(unsigned int reader_index, unsigned int tx_length,
	unsigned char tx_buffer[], unsigned int *rx_length,
	unsigned char rx_buffer[], int protoccol);

RESPONSECODE CCID_Transmit(unsigned int reader_index, unsigned int tx_length,
	const unsigned char tx_buffer[], unsigned short rx_length, unsigned char bBWI);

RESPONSECODE CCID_Receive(unsigned int reader_index,
	/*@out@*/ unsigned int *rx_length,
	/*@out@*/ unsigned char rx_buffer[], unsigned char *chain_parameter);

RESPONSECODE SetParameters(unsigned int reader_index, char protocol,
	unsigned int length, unsigned char buffer[]);

int isCharLevel(int reader_index);

