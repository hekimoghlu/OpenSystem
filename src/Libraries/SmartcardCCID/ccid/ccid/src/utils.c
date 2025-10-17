/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
#include <string.h>
#include <pcsclite.h>

#include <config.h>
#include "ccid.h"
#include "defs.h"
#include "ccid_ifdhandler.h"
#include "utils.h"
#include "debug.h"

int ReaderIndex[CCID_DRIVER_MAX_READERS];
#define FREE_ENTRY -42

void InitReaderIndex(void)
{
	int i;

	for (i=0; i<CCID_DRIVER_MAX_READERS; i++)
		ReaderIndex[i] = FREE_ENTRY;
} /* InitReaderIndex */

int GetNewReaderIndex(const int Lun)
{
	int i;

	/* check that Lun is NOT already used */
	for (i=0; i<CCID_DRIVER_MAX_READERS; i++)
		if (Lun == ReaderIndex[i])
			break;

	if (i < CCID_DRIVER_MAX_READERS)
	{
		DEBUG_CRITICAL2("Lun: %d is already used", Lun);
		return -1;
	}

	for (i=0; i<CCID_DRIVER_MAX_READERS; i++)
		if (FREE_ENTRY == ReaderIndex[i])
		{
			ReaderIndex[i] = Lun;
			return i;
		}

	DEBUG_CRITICAL("ReaderIndex[] is full");
	return -1;
} /* GetReaderIndex */

int LunToReaderIndex(const int Lun)
{
	int i;

	for (i=0; i<CCID_DRIVER_MAX_READERS; i++)
		if (Lun == ReaderIndex[i])
			return i;

	DEBUG_CRITICAL2("Lun: %X not found", Lun);
	return -1;
} /* LunToReaderIndex */

void ReleaseReaderIndex(const int index)
{
	ReaderIndex[index] = FREE_ENTRY;
} /* ReleaseReaderIndex */

/* Read a non aligned 16-bit integer */
uint16_t get_U16(void *buf)
{
	uint16_t value;

	memcpy(&value, buf, sizeof value);

	return value;
}

/* Read a non aligned 32-bit integer */
uint32_t get_U32(void *buf)
{
	uint32_t value;

	memcpy(&value, buf, sizeof value);

	return value;
}

/* Write a non aligned 16-bit integer */
void set_U16(void *buf, uint16_t value)
{
	memcpy(buf, &value, sizeof value);
}

/* Write a non aligned 32-bit integer */
void set_U32(void *buf, uint32_t value)
{
	memcpy(buf, &value, sizeof value);
}

/* swap a 16-bits integer in memory */
/* "AB" -> "BA" */
void p_bswap_16(void *ptr)
{
	uint8_t *array, tmp;

	array = ptr;
	tmp = array[0];
	array[0] = array[1];
	array[1] = tmp;
}

/* swap a 32-bits integer in memory */
/* "ABCD" -> "DCBA" */
void p_bswap_32(void *ptr)
{
	uint8_t *array, tmp;

	array = ptr;
	tmp = array[0];
	array[0] = array[3];
	array[3] = tmp;

	tmp = array[1];
	array[1] = array[2];
	array[2] = tmp;
}
