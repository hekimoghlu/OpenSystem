/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#ifndef _PPS_
#define _PPS_

#include "defines.h"

/*
 * Exported constants definition
 */

#define PPS_OK			0	/* Negotiation OK */
#define PPS_ICC_ERROR		1	/* Comunication error */
#define PPS_HANDSAKE_ERROR	2	/* Agreement not reached */
#define PPS_PROTOCOL_ERROR	3	/* Error starting protocol */
#define PPS_MAX_LENGTH		6

#define PPS_HAS_PPS1(block)	((block[1] & 0x10) == 0x10)
#define PPS_HAS_PPS2(block)	((block[1] & 0x20) == 0x20)
#define PPS_HAS_PPS3(block)	((block[1] & 0x40) == 0x40)

/*
 * Exported data types definition
 */

typedef struct
{
  double f;
  double d;
  double n;
  BYTE t;
}
PPS_ProtocolParameters;

typedef struct
{
  int icc;
  void *protocol;
  PPS_ProtocolParameters parameters;
}
PPS;

/*
 * Exported functions declaration
 */

int PPS_Exchange (int lun, BYTE * params, /*@out@*/ unsigned *length,
	unsigned char *pps1);

#endif /* _PPS_ */

