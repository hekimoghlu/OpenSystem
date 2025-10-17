/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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
#include "crc.h"

/*
 * Generator description
 * ---------------------
 *
 * The CRC algorithm is used to compute a message digest.
 * The polynomial is taken from PGP (parts of the code too).
 */

#define DIGEST_SIZE               (CRCBYTES)
#define CTX_TYPE                  crcword

/*
 * Declarations of internal procedures.
 */

static void MDcrc_Start     _ANSI_ARGS_ ((VOID* context));
static void MDcrc_Update    _ANSI_ARGS_ ((VOID* context, unsigned int character));
static void MDcrc_UpdateBuf _ANSI_ARGS_ ((VOID* context, unsigned char* buffer, int bufLen));
static void MDcrc_Final     _ANSI_ARGS_ ((VOID* context, VOID* digest));

/*
 * Generator definition.
 */

static Trf_MessageDigestDescription mdDescription = {
  "crc",
  sizeof (CTX_TYPE),
  DIGEST_SIZE,
  MDcrc_Start,
  MDcrc_Update,
  MDcrc_UpdateBuf,
  MDcrc_Final,
  NULL
};

/*
 * Additional declarations
 */

static crcword CrcTable [256]; /* THREADING: serialize initialization */

static void
GenCrcLookupTable _ANSI_ARGS_ ((crcword polynomial));

/*
 *------------------------------------------------------*
 *
 *	TrfInit_CRC --
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
TrfInit_CRC (interp)
Tcl_Interp* interp;
{
  GenCrcLookupTable (PRZCRC);

  return Trf_RegisterMessageDigest (interp, &mdDescription);
}

/*
 *------------------------------------------------------*
 *
 *	MDcrc_Start --
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
MDcrc_Start (context)
VOID* context;
{
  /* call md specific initialization here */

  *((crcword*) context) = CRCINIT;
}

/*
 *------------------------------------------------------*
 *
 *	MDcrc_Update --
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
MDcrc_Update (context, character)
VOID* context;
unsigned int   character;
{
  /* call md specific update here */
#define UP(ctx)   ((ctx) << 8)
#define DOWN(ctx) ((ctx) >> CRCSHIFTS)

  crcword       accu;
  unsigned char buf = character;

  accu = *((crcword*) context);
  accu = UP (accu) ^ CrcTable [(unsigned char) (DOWN (accu)) ^ buf];

  *((crcword*) context) = accu;

#undef UP
#undef DOWN
}

/*
 *------------------------------------------------------*
 *
 *	MDcrc_UpdateBuf --
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
MDcrc_UpdateBuf (context, buffer, bufLen)
VOID* context;
unsigned char* buffer;
int   bufLen;
{
  /* call md specific update here */
#define UP(ctx)   ((ctx) << 8)
#define DOWN(ctx) ((ctx) >> CRCSHIFTS)

  crcword accu;
  int     i;

  accu = *((crcword*) context);

  for (i=0; i < bufLen; i++) {
    accu = UP (accu) ^ CrcTable [(unsigned char) (DOWN (accu)) ^ (buffer [i])];
  }

  *((crcword*) context) = accu;

#undef UP
#undef DOWN
}

/*
 *------------------------------------------------------*
 *
 *	MDcrc_Final --
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
MDcrc_Final (context, digest)
VOID* context;
VOID* digest;
{
  /* call md specific finalization here */

  crcword crc = maskcrc (* ((crcword*) context));
  char*   out = (char*) digest;

  /* -*- PGP -*-, was outcrc, BIGENDIAN output */
  /* DEPENDENT on CRCBYTES !!, only a value of 3 is supported here */

  out [0] = (char) ((crc >> 16) & 0xff);
  out [1] = (char) ((crc >>  8) & 0xff);
  out [2] = (char) ((crc >>  0) & 0xff);
  /* -*- PGP -*- */
}

/*
 * Initialize lookup table for crc calculation.
 */

static void
GenCrcLookupTable (poly)
crcword poly;
{
  /* -*- PGP -*-, was 'mk_crctbl' */
  int i;
  crcword t, *p, *q;

  TrfLock; /* THREADING: serialize initialization */

  p = q = CrcTable;

  *q++ = 0;
  *q++ = poly;

  for (i = 1; i < 128; i++)
    {
      t = *++p;
      if (t & CRCHIBIT)
	{
	  t <<= 1;
	  *q++ = t ^ poly;
	  *q++ = t;
	}
      else
	{
	  t <<= 1;
	  *q++ = t;
	  *q++ = t ^ poly;
	}
    }

  TrfUnlock;
  /* -*- PGP -*- */
}
