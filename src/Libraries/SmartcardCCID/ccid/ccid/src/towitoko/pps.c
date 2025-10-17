/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
#include <stdbool.h>
#include "pps.h"
#include "atr.h"
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#include <ifdhandler.h>

#include "commands.h"
#include "defs.h"
#include "debug.h"

/*
 * Not exported funtions declaration
 */

static bool PPS_Match (BYTE * request, unsigned len_request, BYTE * reply, unsigned len_reply);

static unsigned PPS_GetLength (BYTE * block);

static BYTE PPS_GetPCK (BYTE * block, unsigned length);

int
PPS_Exchange (int lun, BYTE * params, unsigned *length, unsigned char *pps1)
{
  BYTE confirm[PPS_MAX_LENGTH];
  unsigned len_request, len_confirm;
  int ret;

  len_request = PPS_GetLength (params);
  params[len_request - 1] = PPS_GetPCK(params, len_request - 1);

  DEBUG_XXD ("PPS: Sending request: ", params, len_request);

  /* Send PPS request */
  if (CCID_Transmit (lun, len_request, params, isCharLevel(lun) ? 4 : 0, 0)
	!= IFD_SUCCESS)
    return PPS_ICC_ERROR;

  /* Get PPS confirm */
  len_confirm = sizeof(confirm);
  if (CCID_Receive (lun, &len_confirm, confirm, NULL) != IFD_SUCCESS)
    return PPS_ICC_ERROR;

  DEBUG_XXD ("PPS: Receiving confirm: ", confirm, len_confirm);

  if (!PPS_Match (params, len_request, confirm, len_confirm))
    ret = PPS_HANDSAKE_ERROR;
  else
    ret = PPS_OK;

  *pps1 = 0x11;	/* default TA1 */

  /* if PPS1 is echoed */
  if (PPS_HAS_PPS1 (params) && PPS_HAS_PPS1 (confirm))
      *pps1 = confirm[2];

  /* Copy PPS handsake */
  memcpy (params, confirm, len_confirm);
  (*length) = len_confirm;

  return ret;
}

static bool
PPS_Match (BYTE * request, unsigned len_request, BYTE * confirm, unsigned len_confirm)
{
  /* See if the reply differs from request */
  if ((len_request == len_confirm) &&	/* same length */
      memcmp (request, confirm, len_request))	/* different contents */
	return false;

  if (len_request < len_confirm)	/* confirm longer than request */
    return false;

  /* See if the card specifies other than default FI and D */
  if ((PPS_HAS_PPS1 (confirm))
		&& (len_confirm > 2)
		&& (confirm[2] != request[2]))
    return false;

  return true;
}

static unsigned
PPS_GetLength (BYTE * block)
{
  unsigned length = 3;

  if (PPS_HAS_PPS1 (block))
    length++;

  if (PPS_HAS_PPS2 (block))
    length++;

  if (PPS_HAS_PPS3 (block))
    length++;

  return length;
}

static BYTE
PPS_GetPCK (BYTE * block, unsigned length)
{
  BYTE pck;
  unsigned i;

  pck = block[0];
  for (i = 1; i < length; i++)
    pck ^= block[i];

  return pck;
}

