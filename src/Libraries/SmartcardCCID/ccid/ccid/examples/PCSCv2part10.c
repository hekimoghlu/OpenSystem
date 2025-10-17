/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 29, 2022.
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
#include <stdio.h>
#include <arpa/inet.h>

#ifdef __APPLE__
#include <PCSC/winscard.h>
#include <PCSC/wintypes.h>
#else
#include <winscard.h>
#endif

#include "PCSCv2part10.h"

int PCSCv2Part10_find_TLV_property_by_tag_from_buffer(
	unsigned char *buffer, int length, int property, int * value_int)
{
	unsigned char *p;
	int found = 0, len;
	int value = -1;
	int ret = -1;	/* not found by default */

	p = buffer;
	while (p-buffer < length)
	{
		if (*p++ == property)
		{
			found = 1;
			break;
		}

		/* go to next tag */
		len = *p++;
		p += len;
	}

	if (found)
	{
		len = *p++;
		ret = 0;

		switch(len)
		{
			case 1:
				value = *p;
				break;
			case 2:
				value = *p + (*(p+1)<<8);
				break;
			case 4:
				value = *p + (*(p+1)<<8) + (*(p+2)<<16) + (*(p+3)<<24);
				break;
			default:
				/* wrong length for an integer */
				ret = -2;
		}
	}

	if (value_int)
		*value_int = value;

	return ret;
} /* PCSCv2Part10_find_TLV_property_by_tag_from_buffer */

int PCSCv2Part10_find_TLV_property_by_tag_from_hcard(SCARDHANDLE hCard,
	int property, int * value)
{
	unsigned char buffer[MAX_BUFFER_SIZE];
	LONG rv;
	DWORD length;
	unsigned int i;
	PCSC_TLV_STRUCTURE *pcsc_tlv;
	DWORD properties_in_tlv_ioctl;
	int found;

	rv = SCardControl(hCard, CM_IOCTL_GET_FEATURE_REQUEST, NULL, 0,
		buffer, sizeof buffer, &length);
	if (rv != SCARD_S_SUCCESS)
		return -1;

	/* get the number of elements instead of the complete size */
	length /= sizeof(PCSC_TLV_STRUCTURE);

	pcsc_tlv = (PCSC_TLV_STRUCTURE *)buffer;
	found = 0;
	for (i = 0; i < length; i++)
	{
		if (FEATURE_GET_TLV_PROPERTIES == pcsc_tlv[i].tag)
		{
			properties_in_tlv_ioctl = ntohl(pcsc_tlv[i].value);
			found = 1;
		}
	}

	if (! found)
		return -3;

	rv= SCardControl(hCard, properties_in_tlv_ioctl, NULL, 0,
		buffer, sizeof buffer, &length);
	if (rv != SCARD_S_SUCCESS)
		return -1;

	return PCSCv2Part10_find_TLV_property_by_tag_from_buffer(buffer,
		length, property, value);
}

