/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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
#include "transformInt.h"
#include "haval.1996/haval.h"

/*
 * Generator description
 * ---------------------
 *
 * The HAVAL alogrithm is used to compute a cryptographically strong
 * message digest.
 */

#define DIGEST_SIZE               (32)
#define CTX_TYPE                  haval_state

/*
 * Declarations of internal procedures.
 */

static void MDHaval_Start     _ANSI_ARGS_ ((VOID* context));
static void MDHaval_Update    _ANSI_ARGS_ ((VOID* context, unsigned int character));
static void MDHaval_UpdateBuf _ANSI_ARGS_ ((VOID* context, unsigned char* buffer, int bufLen));
static void MDHaval_Final     _ANSI_ARGS_ ((VOID* context, VOID* digest));

/*
 * Generator definition.
 */

static Trf_MessageDigestDescription mdDescription = { /* THREADING: constant, read-only => safe */
  "haval",
  sizeof (CTX_TYPE),
  DIGEST_SIZE,
  MDHaval_Start,
  MDHaval_Update,
  MDHaval_UpdateBuf,
  MDHaval_Final,
  NULL
};

/*
 *------------------------------------------------------*
 *
 *	TrfInit_HAVAL --
 *
 *	------------------------------------------------*
 *	Register the generator implemented in this file.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		As of 'Trf_Register'.
 *
 *	Result:
 *		A standard Tcl error code.
 *
 *------------------------------------------------------*
 */

int
TrfInit_HAVAL (interp)
Tcl_Interp* interp;
{
  return Trf_RegisterMessageDigest (interp, &mdDescription);
}

/*
 *------------------------------------------------------*
 *
 *	MDHaval_Start --
 *
 *	------------------------------------------------*
 *	Initialize the internal state of the message
 *	digest generator.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		As of the called procedure.
 *
 *	Result:
 *		None.
 *
 *------------------------------------------------------*
 */

static void
MDHaval_Start (context)
VOID* context;
{
  haval_start ((CTX_TYPE*) context);
}

/*
 *------------------------------------------------------*
 *
 *	MDHaval_Update --
 *
 *	------------------------------------------------*
 *	Update the internal state of the message digest
 *	generator for a single character.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		As of the called procedure.
 *
 *	Result:
 *		None.
 *
 *------------------------------------------------------*
 */

static void
MDHaval_Update (context, character)
VOID* context;
unsigned int   character;
{
  unsigned char buf = character;

  haval_hash ((CTX_TYPE*) context, &buf, 1);
}

/*
 *------------------------------------------------------*
 *
 *	MDHaval_UpdateBuf --
 *
 *	------------------------------------------------*
 *	Update the internal state of the message digest
 *	generator for a character buffer.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		As of the called procedure.
 *
 *	Result:
 *		None.
 *
 *------------------------------------------------------*
 */

static void
MDHaval_UpdateBuf (context, buffer, bufLen)
VOID* context;
unsigned char* buffer;
int   bufLen;
{
  haval_hash ((CTX_TYPE*) context, (unsigned char*) buffer, bufLen);
}

/*
 *------------------------------------------------------*
 *
 *	MDHaval_Final --
 *
 *	------------------------------------------------*
 *	Generate the digest from the internal state of
 *	the message digest generator.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		As of the called procedure.
 *
 *	Result:
 *		None.
 *
 *------------------------------------------------------*
 */

static void
MDHaval_Final (context, digest)
VOID* context;
VOID* digest;
{
  haval_end ((CTX_TYPE*) context, (unsigned char*) digest);
}

/*
 * External code from here on.
 */

#include "haval.1996/haval.c" /* THREADING: import of one constant var, read-only => safe */
