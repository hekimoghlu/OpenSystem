/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
#include <stdio.h>
#include <windows.h>


/*-----------------------------------------------------------------------------
 * TclX_SplitWinCmdLine --
 *   Parse the window command line into arguments.
 *
 * Parameters:
 *   o argcPtr (O) - Count of arguments is returned here.
 *   o argvPtr (O) - Argument vector is returned here.
 * Notes:
 *   This code taken from the Tcl file tclAppInit.c: Copyright (c) 1996 by
 * Sun Microsystems, Inc.
 *-----------------------------------------------------------------------------
 */
void
TclX_SplitWinCmdLine (argcPtr, argvPtr)
    int    *argcPtr;
    char ***argvPtr;
{
    char   *args = GetCommandLine ();
    char **argvlist, *p;
    int size, i;

    /*
     * Precompute an overly pessimistic guess at the number of arguments
     * in the command line by counting non-space spans.
     */
    for (size = 2, p = args; *p != '\0'; p++) {
        if (isspace (*p)) {
            size++;
            while (isspace (*p)) {
                p++;
            }
            if (*p == '\0') {
                break;
            }
        }
    }
    argvlist = (char **) malloc ((unsigned) (size * sizeof (char *)));
    *argvPtr = argvlist;

    /*
     * Parse the Windows command line string.  If an argument begins with a
     * double quote, then spaces are considered part of the argument until the
     * next double quote.  The argument terminates at the second quote.  Note
     * that this is different from the usual Unix semantics.
     */
    for (i = 0, p = args; *p != '\0'; i++) {
        while (isspace (*p)) {
            p++;
        }
        if (*p == '\0') {
            break;
        }
        if (*p == '"') {
            p++;
            (*argvPtr) [i] = p;
            while ((*p != '\0') && (*p != '"')) {
                p++;
            }
        } else {
            (*argvPtr) [i] = p;
            while (*p != '\0' && !isspace(*p)) {
                p++;
            }
        }
        if (*p != '\0') {
            *p = '\0';
            p++;
        }
    }
    (*argvPtr) [i] = NULL;
    *argcPtr = i;
}

/*
 * Concatenate a bunch of files.
 */
int
main (int    argc,
      char **argv)
{
    FILE *fh;
    int idx, c;

    TclX_SplitWinCmdLine (&argc, &argv);


    for (idx = 1; idx < argc; idx++) {
        fh = fopen (argv [idx], "r");
        if (fh == NULL) {
            fprintf (stderr, "error opening \"%s\": %s\n",
                     argv [idx], strerror (errno));
            exit (1);
        }
        while ((c = fgetc (fh)) != EOF) {
            if (fputc (c, stdout) == EOF) {
                fprintf (stderr, "error writing stdout: %s\n", 
                         strerror (errno));
                exit (1);
            }
        }
        if (ferror (fh)) {
            fprintf (stderr, "error reading \"%s\": %s\n",
                     argv [idx], strerror (errno));
            exit (1);
        }
        fclose (fh);
    }
    return 0;
}


