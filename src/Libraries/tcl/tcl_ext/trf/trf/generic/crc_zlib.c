/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 26, 2025.
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

/*
 * Generator description
 * ---------------------
 *
 * The CRC32 algorithm (contained in library 'zlib')
 * is used to compute a message digest.
 */

#define DIGEST_SIZE               4 /* byte == 32 bit */
#define CTX_TYPE                  uLong

/*
 * Declarations of internal procedures.
 */

static void MDcrcz_Start     _ANSI_ARGS_ ((VOID* context));
static void MDcrcz_Update    _ANSI_ARGS_ ((VOID* context, unsigned int character));
static void MDcrcz_UpdateBuf _ANSI_ARGS_ ((VOID* context, unsigned char* buffer, int bufLen));
static void MDcrcz_Final     _ANSI_ARGS_ ((VOID* context, VOID* digest));
static int  MDcrcz_Check     _ANSI_ARGS_ ((Tcl_Interp* interp));

/*
 * Generator definition.
 */

static Trf_MessageDigestDescription mdDescription = {
  "crc-zlib",
  sizeof (CTX_TYPE),
  DIGEST_SIZE,
  MDcrcz_Start,
  MDcrcz_Update,
  MDcrcz_UpdateBuf,
  MDcrcz_Final,
  MDcrcz_Check
};

#define CRC (*((uLong*) context))

/*
 *------------------------------------------------------*
 *
 *	TrfInit_CRC_zlib --
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
TrfInit_CRC_ZLIB (interp)
Tcl_Interp* interp;
{
  return Trf_RegisterMessageDigest (interp, &mdDescription);
}

/*
 *------------------------------------------------------*
 *
 *	MDcrcz_Start --
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
MDcrcz_Start (context)
VOID* context;
{
  /* call md specific initialization here */

  CRC = zf.zcrc32 (0L, Z_NULL, 0);
}

/*
 *------------------------------------------------------*
 *
 *	MDcrcz_Update --
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
MDcrcz_Update (context, character)
VOID* context;
unsigned int   character;
{
  /* call md specific update here */

  unsigned char buf = character;

  CRC = zf.zcrc32 (CRC, &buf, 1);
}

/*
 *------------------------------------------------------*
 *
 *	MDcrcz_UpdateBuf --
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
MDcrcz_UpdateBuf (context, buffer, bufLen)
VOID* context;
unsigned char* buffer;
int   bufLen;
{
  /* call md specific update here */

  CRC = zf.zcrc32 (CRC, buffer, bufLen);
}

/*
 *------------------------------------------------------*
 *
 *	MDcrcz_Final --
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
MDcrcz_Final (context, digest)
VOID* context;
VOID* digest;
{
  /* call md specific finalization here */

  uLong crc = CRC;
  char*   out = (char*) digest;

  /* LITTLE ENDIAN output */
  out [3] = (char) ((crc >> 24) & 0xff);
  out [2] = (char) ((crc >> 16) & 0xff);
  out [1] = (char) ((crc >>  8) & 0xff);
  out [0] = (char) ((crc >>  0) & 0xff);
}

/*
 *------------------------------------------------------*
 *
 *	MDcrcz_Check --
 *
 *	------------------------------------------------*
 *	Check for existence of libz, load it.
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

static int
MDcrcz_Check (interp)
Tcl_Interp* interp;
{
  return TrfLoadZlib (interp);
}

