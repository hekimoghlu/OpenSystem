/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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
#include "loadman.h"

/*
 * Generator description
 * ---------------------
 *
 * The SHA1 alogrithm is used to compute a cryptographically strong
 * message digest.
 */

#ifndef OTP
#define DIGEST_SIZE               (SHA_DIGEST_LENGTH)
#else
#define DIGEST_SIZE               (8)
#endif
#define CTX_TYPE                  SHA_CTX

/*
 * Declarations of internal procedures.
 */

static void MDsha1_Start     _ANSI_ARGS_ ((VOID* context));
static void MDsha1_Update    _ANSI_ARGS_ ((VOID* context, unsigned int character));
static void MDsha1_UpdateBuf _ANSI_ARGS_ ((VOID* context, unsigned char* buffer, int bufLen));
static void MDsha1_Final     _ANSI_ARGS_ ((VOID* context, VOID* digest));
static int  MDsha1_Check     _ANSI_ARGS_ ((Tcl_Interp* interp));

/*
 * Generator definition.
 */

static Trf_MessageDigestDescription mdDescription = { /* THREADING: constant, read-only => safe */
#ifndef OTP
  "sha1",
#else
  "otp_sha1",
#endif
  sizeof (CTX_TYPE),
  DIGEST_SIZE,
  MDsha1_Start,
  MDsha1_Update,
  MDsha1_UpdateBuf,
  MDsha1_Final,
  MDsha1_Check
};

/*
 *------------------------------------------------------*
 *
 *	TrfInit_SHA1 --
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
#ifndef OTP
TrfInit_SHA1 (interp)
#else
TrfInit_OTP_SHA1 (interp)
#endif
Tcl_Interp* interp;
{
  return Trf_RegisterMessageDigest (interp, &mdDescription);
}

/*
 *------------------------------------------------------*
 *
 *	MDsha1_Start --
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
MDsha1_Start (context)
VOID* context;
{
  sha1f.init ((SHA_CTX*) context);
}

/*
 *------------------------------------------------------*
 *
 *	MDsha1_Update --
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
MDsha1_Update (context, character)
VOID* context;
unsigned int   character;
{
  unsigned char buf = character;

  sha1f.update ((SHA_CTX*) context, &buf, 1);
}

/*
 *------------------------------------------------------*
 *
 *	MDsha1_UpdateBuf --
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
MDsha1_UpdateBuf (context, buffer, bufLen)
VOID* context;
unsigned char* buffer;
int   bufLen;
{
  sha1f.update ((SHA_CTX*) context, (unsigned char*) buffer, bufLen);
}

/*
 *------------------------------------------------------*
 *
 *	MDsha1_Final --
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
MDsha1_Final (context, digest)
VOID* context;
VOID* digest;
{
#ifndef OTP
  sha1f.final ((unsigned char*) digest, (SHA_CTX*) context);
#else
    unsigned int result[SHA_DIGEST_LENGTH / sizeof (char)];

    sha1f.final ((unsigned char*) result, (SHA_CTX*) context);

    result[0] ^= result[2];
    result[1] ^= result[3];
    result[0] ^= result[4];

    Trf_FlipRegisterLong ((VOID*) result, DIGEST_SIZE);
    memcpy ((VOID *) digest, (VOID *) result, DIGEST_SIZE);
#endif
}

/*
 *------------------------------------------------------*
 *
 *	MDsha1_Check --
 *
 *	------------------------------------------------*
 *	Do global one-time initializations of the message
 *	digest generator.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		Loads the shared library containing the
 *		SHA1 functionality
 *
 *	Result:
 *		A standard Tcl error code.
 *
 *------------------------------------------------------*
 */

static int
MDsha1_Check (interp)
Tcl_Interp* interp;
{
  return TrfLoadSHA1 (interp);
}
