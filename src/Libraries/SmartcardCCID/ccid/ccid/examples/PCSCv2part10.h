/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 17, 2024.
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
#ifndef __reader_h__

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_READER_H
#include <reader.h>
#else

/**
 * Provide source compatibility on different platforms
 */
#define SCARD_CTL_CODE(code) (0x42000000 + (code))

/**
 * PC/SC part 10 v2.02.07 March 2010 reader tags
 */
#define CM_IOCTL_GET_FEATURE_REQUEST SCARD_CTL_CODE(3400)

#define FEATURE_GET_TLV_PROPERTIES       0x12	/**< Get TLV properties */

#include <inttypes.h>

/* Set structure elements aligment on bytes
 * http://gcc.gnu.org/onlinedocs/gcc/Structure_002dPacking-Pragmas.html */
#if defined(__APPLE__) | defined(sun)
#pragma pack(1)
#else
#pragma pack(push, 1)
#endif

/** the structure must be 6-bytes long */
typedef struct
{
	uint8_t tag; /**< Tag */
	uint8_t length; /**< Length */
	uint32_t value;	/**< This value is always in BIG ENDIAN format as documented in PCSC v2 part 10 ch 2.2 page 2. You can use ntohl() for example */
} PCSC_TLV_STRUCTURE;

/* restore default structure elements alignment */
#if defined(__APPLE__) | defined(sun)
#pragma pack()
#else
#pragma pack(pop)
#endif

/* properties returned by FEATURE_GET_TLV_PROPERTIES */
#define PCSCv2_PART10_PROPERTY_wLcdLayout 1		/**< wLcdLayout */
#define PCSCv2_PART10_PROPERTY_bEntryValidationCondition 2	/**< bEntryValidationCondition */
#define PCSCv2_PART10_PROPERTY_bTimeOut2 3	/**< bTimeOut2 */
#define PCSCv2_PART10_PROPERTY_wLcdMaxCharacters 4 /**< wLcdMaxCharacters */
#define PCSCv2_PART10_PROPERTY_wLcdMaxLines 5 /**< wLcdMaxLines */
#define PCSCv2_PART10_PROPERTY_bMinPINSize 6 /**< bMinPINSize */
#define PCSCv2_PART10_PROPERTY_bMaxPINSize 7 /**< bMaxPINSize */
#define PCSCv2_PART10_PROPERTY_sFirmwareID 8 /**< sFirmwareID */
#define PCSCv2_PART10_PROPERTY_bPPDUSupport 9 /**< bPPDUSupport */
#define PCSCv2_PART10_PROPERTY_dwMaxAPDUDataSize 10 /**< dwMaxAPDUDataSize */
#define PCSCv2_PART10_PROPERTY_wIdVendor 11 /**< wIdVendor */
#define PCSCv2_PART10_PROPERTY_wIdProduct 12 /**< wIdProduct */

#endif
#endif

/**
 * @file
 * @defgroup API API
 *
 * The available PC/SC v2 part 10 tags are (from pcsc-lite 1.8.5):
 *
 * - \ref PCSCv2_PART10_PROPERTY_wLcdLayout
 * - \ref PCSCv2_PART10_PROPERTY_bEntryValidationCondition
 * - \ref PCSCv2_PART10_PROPERTY_bTimeOut2
 * - \ref PCSCv2_PART10_PROPERTY_wLcdMaxCharacters
 * - \ref PCSCv2_PART10_PROPERTY_wLcdMaxLines
 * - \ref PCSCv2_PART10_PROPERTY_bMinPINSize
 * - \ref PCSCv2_PART10_PROPERTY_bMaxPINSize
 * - \ref PCSCv2_PART10_PROPERTY_sFirmwareID
 * - \ref PCSCv2_PART10_PROPERTY_bPPDUSupport
 * - \ref PCSCv2_PART10_PROPERTY_dwMaxAPDUDataSize
 * - \ref PCSCv2_PART10_PROPERTY_wIdVendor
 * - \ref PCSCv2_PART10_PROPERTY_wIdProduct
 *
 * Example of code:
 * @include sample.c
 */

/**
 * @brief Find an integer value by tag from TLV buffer
 * @ingroup API
 *
 * @param buffer buffer received from FEATURE_GET_TLV_PROPERTIES
 * @param length buffer length
 * @param property tag searched
 * @param[out] value value found
 * @return Error code
 *
 * @retval 0 success
 * @retval -1 not found
 * @retval -2 invalid length in the TLV
 *
 */
int PCSCv2Part10_find_TLV_property_by_tag_from_buffer(
	unsigned char *buffer, int length, int property, int * value);

/**
 * @brief Find a integer value by tag from a PC/SC card handle
 * @ingroup API
 *
 * @param hCard card handle as returned by SCardConnect()
 * @param property tag searched
 * @param[out] value value found
 * @return Error code (see PCSCv2Part10_find_TLV_property_by_tag_from_buffer())
 */
int PCSCv2Part10_find_TLV_property_by_tag_from_hcard(SCARDHANDLE hCard,
	int property, int * value);

