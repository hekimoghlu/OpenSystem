/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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
/* Don't use stubs since we are in the main application. */
#undef USE_TCL_STUBS

#include "expect_cf.h"
#include <stdio.h>
#include "tcl.h"
#include "expect_tcl.h"
#include <stdlib.h>

int
main(argc, argv)
int argc;
char *argv[];
{
	int rc = 0;
	char buffer [30];

	Tcl_Interp *interp = Tcl_CreateInterp();
	Tcl_FindExecutable(argv[0]);

	if (Tcl_Init(interp) == TCL_ERROR) {
	    fprintf(stderr,"Tcl_Init failed: %s\n",Tcl_GetStringResult (interp));
	    (void) exit(1);
	}

	if (Expect_Init(interp) == TCL_ERROR) {
	    fprintf(stderr,"Expect_Init failed: %s\n",Tcl_GetStringResult (interp));
	    (void) exit(1);
	}

	exp_parse_argv(interp,argc,argv);

	/* become interactive if requested or "nothing to do" */
	if (exp_interactive)
		(void) exp_interpreter(interp,(Tcl_Obj *)0);
	else if (exp_cmdfile)
		rc = exp_interpret_cmdfile(interp,exp_cmdfile);
	else if (exp_cmdfilename)
		rc = exp_interpret_cmdfilename(interp,exp_cmdfilename);

	/* assert(exp_cmdlinecmds != 0) */

	/* SF #439042 -- Allow overide of "exit" by user / script
	 */

	sprintf(buffer, "exit %d", rc);
	Tcl_Eval(interp, buffer); 
	/*NOTREACHED*/
	return 0;		/* Needed only to prevent compiler warning. */
}

