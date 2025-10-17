/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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

extern TrfStubs trfStubs;


/*
 *------------------------------------------------------*
 *
 *	Trf_Init --
 *
 *	------------------------------------------------*
 *	Standard procedure required by 'load'. 
 *	Initializes this extension.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		As of 'TrfGetRegistry'.
 *
 *	Result:
 *		A standard Tcl error code.
 *
 *------------------------------------------------------*
 */

int
Trf_Init (interp)
Tcl_Interp* interp;
{
  Trf_Registry*  registry;
  int            res;

#ifdef USE_TCL_STUBS
  CONST char* actualVersion;

  actualVersion = Tcl_InitStubs(interp, "8.1", 0);
  if (actualVersion == NULL) {
    return TCL_ERROR;
  }
#endif

  if (Trf_IsInitialized (interp)) {
      /*
       * catch multiple initialization of an interpreter
       */
      return TCL_OK;
    }

  registry = TrfGetRegistry (interp);

  if (!registry) {
    return TCL_ERROR;
  }

#ifdef USE_TCL_STUBS
  /*
   * Discern which variant of stacked channels is or can be in use
   * by the core which loaded us.
   */

  {
    int major, minor, patchlevel, releasetype;
    Tcl_GetVersion (&major, &minor, &patchlevel, &releasetype);

    if (major > 8) {
      /* Beyond 8.3.2 */
      registry->patchVariant = PATCH_832;
    } else if (major == 8) {
      if ((minor > 3) ||
	  ((minor == 3) && (patchlevel > 1) &&
	   (releasetype == TCL_FINAL_RELEASE))) {
	/* Is 8.3.2 or beyond */
	registry->patchVariant = PATCH_832;
      } else if (minor > 1) {
	/* Is 8.2 or beyond */
	registry->patchVariant = PATCH_82;
      } else {
	/* 8.0.x or 8.1.x */
	registry->patchVariant = PATCH_ORIG;
      }
    } else /* major < 8 */ {
      Tcl_AppendResult (interp,
			"Cannot this compilation of Trf with a core below 8.0",
			(char*) NULL);
      return TCL_ERROR;
    }
  }
#endif

  /*
   * Register us as a now available package
   */
  
  PROVIDE (interp, trfStubs);
  res = TrfInit_Unstack (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_Info (interp);

  if (res != TCL_OK)
    return res;
  
#ifdef ENABLE_BINIO
  res = TrfInit_Binio (interp);

  if (res != TCL_OK)
    return res;
#endif

  /*
   * Register error correction algorithms.
   */

  res = TrfInit_RS_ECC (interp);
  
  if (res != TCL_OK)
    return res;

  /*
   * Register compressors.
   */

  res = TrfInit_ZIP (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_BZ2 (interp);

  if (res != TCL_OK)
    return res;

  /*
   * Register message digests
   */

  res = TrfInit_CRC (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_ADLER (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_CRC_ZLIB (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_MD5 (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_OTP_MD5 (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_MD2 (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_HAVAL (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_SHA (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_SHA1 (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_OTP_SHA1 (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_RIPEMD160 (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_RIPEMD128 (interp);

  if (res != TCL_OK)
    return res;

  /*
   * Register freeform transformation, reflector into tcl level
   */

  res = TrfInit_Transform (interp);

  if (res != TCL_OK)
    return res;
  
  /*
   * Register crypt commands for pwd auth.
   */

  res = TrfInit_Crypt (interp);

  if (res != TCL_OK)
    return res;

  /*
   * Register standard encodings.
   */

  res = TrfInit_Ascii85 (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_UU (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_B64 (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_Bin (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_Oct (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_OTP_WORDS (interp);

  if (res != TCL_OK)
    return res;

  res = TrfInit_QP (interp);

  if (res != TCL_OK)
    return res;

  return TrfInit_Hex (interp);
}

/*
 *------------------------------------------------------*
 *
 *	Trf_SafeInit --
 *
 *	------------------------------------------------*
 *	Standard procedure required by 'load'. 
 *	Initializes this extension for a safe interpreter.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		As of 'TrfGetRegistry'
 *
 *	Result:
 *		A standard Tcl error code.
 *
 *------------------------------------------------------*
 */

int
Trf_SafeInit (interp)
Tcl_Interp* interp;
{
  return Trf_Init (interp);
}

/*
 *------------------------------------------------------*
 *
 *	Trf_IsInitialized --
 *
 *	------------------------------------------------*
 *	Checks, wether the extension is initialized in
 *	the specified interpreter.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		None.
 *
 *	Result:
 *		1 if and onlly if the extension is already
 *		initialized in the specified interpreter,
 *		0 else.
 *
 *------------------------------------------------------*
 */

int
Trf_IsInitialized (interp)
Tcl_Interp* interp;
{
  Trf_Registry* registry;

  registry = TrfPeekForRegistry (interp);

  return registry != (Trf_Registry*) NULL;
}

#if GT81 && defined (TCL_THREADS) /* THREADING: lock procedures */
/*
 *------------------------------------------------------*
 *
 *	Trf(Un)LockIt --
 *
 *	------------------------------------------------*
 *	Internal functions, used to serialize write-access
 *	to several global variables. Required only for
 *	a thread-enabled Tcl 8.1.x and beyond.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		None.
 *
 *	Result:
 *		None.
 *
 *------------------------------------------------------*
 */

TCL_DECLARE_MUTEX(trfInitMutex)

void
TrfLockIt ()
{
  Tcl_MutexLock (&trfInitMutex);
}

void
TrfUnlockIt ()
{
  Tcl_MutexUnlock (&trfInitMutex);
}

#endif /* GT81 */

