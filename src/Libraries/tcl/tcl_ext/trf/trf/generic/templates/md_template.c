/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
#include "xxx/xxx.h"

/*
 * Generator description
 * ---------------------
 *
 * The XXX alogrithm is used to compute a cryptographically strong
 * message digest.
 */

#define DIGEST_SIZE               (0)
#define CTX_TYPE                  XXX

/*
 * Declarations of internal procedures.
 */

static void MD_Start     _ANSI_ARGS_ ((VOID* context));
static void MD_Update    _ANSI_ARGS_ ((VOID* context, int character));
static void MD_UpdateBuf _ANSI_ARGS_ ((VOID* context, char* buffer, int bufLen));
static void MD_Final     _ANSI_ARGS_ ((VOID* context, VOID* digest));
static int  MD_Check     _ANSI_ARGS_ ((Tcl_Interp* interp));

/*
 * Generator definition.
 */

static Trf_MessageDigestDescription mdDescription = {
  "xxx",
  sizeof (CTX_TYPE),
  DIGEST_SIZE,
  MD_Start,
  MD_Update,
  MD_UpdateBuf,
  MD_Final,
  MD_Check
};

/*
 *------------------------------------------------------*
 *
 *	TrfInit_XXX --
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
TrfInit_XXX (interp)
Tcl_Interp* interp;
{
  return Trf_RegisterMessageDigest (interp, &mdDescription);
}

/*
 *------------------------------------------------------*
 *
 *	MD_Start --
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
MD_Start (context)
VOID* context;
{
  /* call md specific initialization here */
}

/*
 *------------------------------------------------------*
 *
 *	MD_Update --
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
MD_Update (context, character)
VOID* context;
int   character;
{
  char buf = character;

  /* call md specific update here */
}

/*
 *------------------------------------------------------*
 *
 *	MD_UpdateBuf --
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
MD_Updatebuf (context, buffer, bufLen)
VOID* context;
char* buffer;
int   bufLen;
{
  sha_trf_info* s = context;

  if ((s->count + bufLen) < CHUNK_SIZE) {
    /*
     * Not enough for full chunk. Remember incoming
     * data and wait for another call containing more data.
     */

    memcpy ((VOID*) (s->buf + s->count), (VOID*) buffer, bufLen);
    s->count += bufLen;
  } else {
    /*
     * Complete chunk with incoming data, update digest,
     * then use all chunks contained in the buffer. Remember
     * an incomplete chunk and wait for further calls.
     */

    int k = CHUNK_SIZE - s->count;

    if (k < CHUNK_SIZE) {
      memcpy ((VOID*) (s->buf + s->count), (VOID*) buffer, k);

      sha_update (&s->s, s->buf, CHUNK_SIZE);

      buffer += k;
      bufLen -= k;
    } /* k == CHUNK_SIZE => internal buffer was empty, so skip it entirely */

    while (bufLen > CHUNK_SIZE) {
      sha_update (&s->s, buffer, CHUNK_SIZE);

      buffer += CHUNK_SIZE;
      bufLen -= CHUNK_SIZE;
    }

    s->count = bufLen;
    if (bufLen > 0)
      memcpy ((VOID*) s->buf, (VOID*) buffer, bufLen);
  }
}

/*
 *------------------------------------------------------*
 *
 *	MD_Final --
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
MD_Final (context, digest)
VOID* context;
VOID* digest;
{
  /* call md specific finalization here */
}

/*
 *------------------------------------------------------*
 *
 *	MD_Check --
 *
 *	------------------------------------------------*
 *	Check environment (required shared libs, ...).
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
MD_Check (interp)
Tcl_Interp* interp;
{
  return TCL_OK;
}

/*
 * External code from here on.
 */

#include "xxx/xxx.c"
