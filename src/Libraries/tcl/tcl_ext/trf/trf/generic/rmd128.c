/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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
#include "ripemd/rmd128.h"

/*
 * Generator description
 * ---------------------
 *
 * The RIPEMD-128 alogrithm is used to compute a cryptographically strong
 * message digest.
 */

#define DIGEST_SIZE   (16)
/*#define CTX_TYPE                   */
#define CONTEXT_SIZE  (16)
#define CHUNK_SIZE    (64)

typedef struct ripemd_context {
  dword state [5];		/* state variables of ripemd-128 */
  byte  buf   [CHUNK_SIZE];	/* buffer of 16-dword's          */
  byte  byteCount;		/* number of bytes in buffer     */
  dword lowc;			/* lower half of a 64bit counter */
  dword highc;			/* upper half of a 64bit counter */
} ripemd_context;


/*
 * Declarations of internal procedures.
 */

static void MDrmd128_Start     _ANSI_ARGS_ ((VOID* context));
static void MDrmd128_Update    _ANSI_ARGS_ ((VOID* context, unsigned int character));
static void MDrmd128_UpdateBuf _ANSI_ARGS_ ((VOID* context, unsigned char* buffer, int bufLen));
static void MDrmd128_Final     _ANSI_ARGS_ ((VOID* context, VOID* digest));
static void CountLength  _ANSI_ARGS_ ((ripemd_context* ctx,
				       unsigned int    nbytes));

/*
 * Generator definition.
 */

static Trf_MessageDigestDescription mdDescription = { /* THREADING: constant, read-only => safe */
  "ripemd128",
  sizeof (ripemd_context),
  DIGEST_SIZE,
  MDrmd128_Start,
  MDrmd128_Update,
  MDrmd128_UpdateBuf,
  MDrmd128_Final,
  NULL
};

/*
 *------------------------------------------------------*
 *
 *	TrfInit_RIPEMD128 --
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
TrfInit_RIPEMD128 (interp)
Tcl_Interp* interp;
{
  return Trf_RegisterMessageDigest (interp, &mdDescription);
}

/*
 *------------------------------------------------------*
 *
 *	MDrmd128_Start --
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
MDrmd128_Start (context)
VOID* context;
{
  ripemd_context* ctx = (ripemd_context*) context;

  ripemd128_MDinit (ctx->state);
  memset (ctx->buf, '\0', CHUNK_SIZE);

  ctx->byteCount = 0;
  ctx->lowc     = 0;
  ctx->highc    = 0;
}

/*
 *------------------------------------------------------*
 *
 *	MDrmd128_Update --
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
MDrmd128_Update (context, character)
VOID* context;
unsigned int   character;
{
  ripemd_context* ctx = (ripemd_context*) context;

  ctx->buf [ctx->byteCount] = character;
  ctx->byteCount ++;

  if (ctx->byteCount == CHUNK_SIZE) {
    CountLength (ctx, CHUNK_SIZE);

#ifdef WORDS_BIGENDIAN
    Trf_FlipRegisterLong (ctx->buf, CHUNK_SIZE);
#endif
    ripemd128_compress (ctx->state, (dword*) ctx->buf);
    ctx->byteCount = 0;
  }
}

/*
 *------------------------------------------------------*
 *
 *	MDrmd128_UpdateBuf --
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
MDrmd128_UpdateBuf (context, buffer, bufLen)
VOID* context;
unsigned char* buffer;
int   bufLen;
{
  ripemd_context* ctx = (ripemd_context*) context;

  if ((ctx->byteCount + bufLen) < CHUNK_SIZE) {
    /*
     * Not enough for full chunk. Remember incoming
     * data and wait for another call containing more data.
     */

    memcpy ((VOID*) (ctx->buf + ctx->byteCount), (VOID*) buffer, bufLen);
    ctx->byteCount += bufLen;
  } else {
    /*
     * Complete chunk with incoming data, update digest,
     * then use all chunks contained in the buffer. Remember
     * an incomplete chunk and wait for further calls.
     */

    int k = CHUNK_SIZE - ctx->byteCount;

    if (k < CHUNK_SIZE) {
      memcpy ((VOID*) (ctx->buf + ctx->byteCount), (VOID*) buffer, k);

      CountLength (ctx, CHUNK_SIZE);

#ifdef WORDS_BIGENDIAN
      Trf_FlipRegisterLong (ctx->buf, CHUNK_SIZE);
#endif
      ripemd128_compress (ctx->state, (dword*) ctx->buf);

      buffer += k;
      bufLen -= k;
    } /* k == CHUNK_SIZE => internal buffer was empty, so skip it entirely */

    while (bufLen >= CHUNK_SIZE) {
      CountLength (ctx, CHUNK_SIZE);

#ifdef WORDS_BIGENDIAN
      Trf_FlipRegisterLong (buffer, CHUNK_SIZE);
#endif
      ripemd128_compress (ctx->state, (dword*) buffer);
#ifdef WORDS_BIGENDIAN
      Trf_FlipRegisterLong (buffer, CHUNK_SIZE);
#endif

      buffer += CHUNK_SIZE;
      bufLen -= CHUNK_SIZE;
    }

    ctx->byteCount = bufLen;
    if (bufLen > 0) {
      memcpy ((VOID*) ctx->buf, (VOID*) buffer, bufLen);
    }
  }
}

/*
 *------------------------------------------------------*
 *
 *	MDrmd128_Final --
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
MDrmd128_Final (context, digest)
VOID* context;
VOID* digest;
{
  ripemd_context* ctx = (ripemd_context*) context;

  CountLength (ctx, ctx->byteCount);

  ripemd128_MDfinish (ctx->state, ctx->buf, ctx->lowc, ctx->highc);

  memcpy (digest, ctx->state, DIGEST_SIZE);
#ifdef WORDS_BIGENDIAN
  Trf_FlipRegisterLong (digest, DIGEST_SIZE);
#endif
}

/*
 *------------------------------------------------------*
 *
 *	CountLength --
 *
 *	------------------------------------------------*
 *	Update the 64bit counter in the context structure
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		See above.
 *
 *	Result:
 *		None.
 *
 *------------------------------------------------------*
 */

static void
CountLength (ctx, nbytes)
     ripemd_context* ctx;
     unsigned int    nbytes;
{
  /* update length counter */

  if ((ctx->lowc + nbytes) < ctx->lowc) {
    /* overflow to msb of length */
    ctx->highc ++;
  }

  ctx->lowc += nbytes;
}

/*
 * External code from here on.
 */

#include "ripemd/rmd128.c"
