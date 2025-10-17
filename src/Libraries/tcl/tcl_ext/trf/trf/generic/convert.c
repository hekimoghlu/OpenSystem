/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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
 * forward declarations of all internally used procedures.
 */

static Trf_Options
CreateOptions _ANSI_ARGS_ ((ClientData clientData));

static void
DeleteOptions _ANSI_ARGS_ ((Trf_Options options,
			    ClientData  clientData));
static int
CheckOptions  _ANSI_ARGS_ ((Trf_Options            options,
			    Tcl_Interp*            interp,
			    CONST Trf_BaseOptions* baseOptions,
			    ClientData             clientData));
static int
SetOption     _ANSI_ARGS_ ((Trf_Options    options,
			    Tcl_Interp*    interp,
			    CONST char*    optname,
			    CONST Tcl_Obj* optvalue,
			    ClientData     clientData));
static int
QueryOptions  _ANSI_ARGS_ ((Trf_Options options,
			    ClientData  clientData));

static void
SeekQueryOptions  _ANSI_ARGS_ ((Tcl_Interp*          interp,
				Trf_Options          options,
				Trf_SeekInformation* seekInfo,
				ClientData           clientData));


/*
 *------------------------------------------------------*
 *
 *	Trf_ConverterOptions --
 *
 *	------------------------------------------------*
 *	Accessor to the set of vectors realizing option
 *	processing for converter procedures.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		None.
 *
 *	Result:
 *		See above.
 *
 *------------------------------------------------------*
 */

Trf_OptionVectors*
Trf_ConverterOptions ()
{
  static Trf_OptionVectors optVec = /* THREADING: constant, read-only => safe */
    {
      CreateOptions,
      DeleteOptions,
      CheckOptions,
      NULL,      /* no string procedure for 'SetOption' */
      SetOption,
      QueryOptions,
      SeekQueryOptions
    };

  return &optVec;
}

/*
 *------------------------------------------------------*
 *
 *	CreateOptions --
 *
 *	------------------------------------------------*
 *	Create option structure for converter transformations.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		Allocates memory and initializes it as
 *		option structure for converter
 *		transformations.
 *
 *	Result:
 *		A reference to the allocated block of
 *		memory.
 *
 *------------------------------------------------------*
 */

static Trf_Options
CreateOptions (clientData)
ClientData clientData;
{
  Trf_ConverterOptionBlock* o;

  o = (Trf_ConverterOptionBlock*) ckalloc (sizeof (Trf_ConverterOptionBlock));
  o->mode = TRF_UNKNOWN_MODE;

  return (Trf_Options) o;
}

/*
 *------------------------------------------------------*
 *
 *	DeleteOptions --
 *
 *	------------------------------------------------*
 *	Delete option structure of a converter transformations
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		A memory block allocated by 'CreateOptions'
 *		is released.
 *
 *	Result:
 *		None.
 *
 *------------------------------------------------------*
 */

static void
DeleteOptions (options, clientData)
Trf_Options options;
ClientData  clientData;
{
  Trf_ConverterOptionBlock* o = (Trf_ConverterOptionBlock*) options;
  ckfree ((VOID*) o);
}

/*
 *------------------------------------------------------*
 *
 *	CheckOptions --
 *
 *	------------------------------------------------*
 *	Check the given option structure for errors.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		May modify the given structure to set
 *		default values into uninitialized parts.
 *
 *	Result:
 *		A standard Tcl error code.
 *
 *------------------------------------------------------*
 */

static int
CheckOptions (options, interp, baseOptions, clientData)
Trf_Options            options;
Tcl_Interp*            interp;
CONST Trf_BaseOptions* baseOptions;
ClientData             clientData;
{
  Trf_ConverterOptionBlock* o = (Trf_ConverterOptionBlock*) options;

  if (baseOptions->attach == (Tcl_Channel) NULL) /* IMMEDIATE? */ {
    if (o->mode == TRF_UNKNOWN_MODE) {
      Tcl_AppendResult (interp, "-mode option not set", (char*) NULL);
      return TCL_ERROR;
    }
  } else /* ATTACH */ {
    if (o->mode == TRF_UNKNOWN_MODE) {
      o->mode = TRF_ENCODE_MODE;
    }
  }

  return TCL_OK;
}

/*
 *------------------------------------------------------*
 *
 *	SetOption --
 *
 *	------------------------------------------------*
 *	Define value of given option.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		Sets the given value into the option
 *		structure
 *
 *	Result:
 *		A standard Tcl error code.
 *
 *------------------------------------------------------*
 */

static int
SetOption (options, interp, optname, optvalue, clientData)
Trf_Options options;
Tcl_Interp* interp;
CONST char* optname;
CONST Tcl_Obj* optvalue;
ClientData  clientData;
{
  Trf_ConverterOptionBlock* o = (Trf_ConverterOptionBlock*) options;
  int                     len;
  CONST char*             value;

  len = strlen (optname+1);

  switch (optname [1]) {
  case 'm':
    if (0 != strncmp (optname, "-mode", len))
      goto unknown_option;

    value = Tcl_GetStringFromObj ((Tcl_Obj*) optvalue, NULL);
    len   = strlen (value);

    switch (value [0]) {
    case 'e':
      if (0 != strncmp (value, "encode", len))
	goto unknown_mode;
      
      o->mode = TRF_ENCODE_MODE;
      break;

    case 'd':
      if (0 != strncmp (value, "decode", len))
	goto unknown_mode;
      
      o->mode = TRF_DECODE_MODE;
      break;

    default:
    unknown_mode:
      Tcl_AppendResult (interp, "unknown mode '", (char*) NULL);
      Tcl_AppendResult (interp, value, (char*) NULL);
      Tcl_AppendResult (interp, "', should be 'encode' or 'decode'", (char*) NULL);
      return TCL_ERROR;
      break;
    } /* switch optvalue */
    break;

  default:
    goto unknown_option;
    break;
  }

  return TCL_OK;

 unknown_option:
  Tcl_AppendResult (interp, "unknown option '", (char*) NULL);
  Tcl_AppendResult (interp, optname, (char*) NULL);
  Tcl_AppendResult (interp, "', should be '-mode'", (char*) NULL);
  return TCL_ERROR;
}

/*
 *------------------------------------------------------*
 *
 *	QueryOptions --
 *
 *	------------------------------------------------*
 *	Returns a value indicating wether the encoder or
 *	decoder set of vectors is to be used by immediate
 *	execution.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		None
 *
 *	Result:
 *		1 - use encoder vectors.
 *		0 - use decoder vectors.
 *
 *------------------------------------------------------*
 */

static int
QueryOptions (options, clientData)
Trf_Options options;
ClientData clientData;
{
  Trf_ConverterOptionBlock* o = (Trf_ConverterOptionBlock*) options;

  return (o->mode == TRF_ENCODE_MODE ? 1 : 0);
}

/*
 *------------------------------------------------------*
 *
 *	SeekQueryOptions --
 *
 *	------------------------------------------------*
 *	Modifies the natural seek policy according to the
 *	configuration of the transformation.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		May modify 'seekInfo'.
 *
 *	Result:
 *		None.
 *
 *------------------------------------------------------*
 */

static void
SeekQueryOptions (interp, options, seekInfo, clientData)
     Tcl_Interp*          interp;
     Trf_Options          options;
     Trf_SeekInformation* seekInfo;
     ClientData           clientData;
{
  Trf_ConverterOptionBlock* o = (Trf_ConverterOptionBlock*) options;

  if (o->mode == TRF_DECODE_MODE) {
    /* The conversion is backward, swap the seek information.
     */

    int t                       = seekInfo->numBytesTransform;
    seekInfo->numBytesTransform = seekInfo->numBytesDown;
    seekInfo->numBytesDown      = t;
  }
}
