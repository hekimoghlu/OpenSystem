/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
#include "zsh.mdh"
#include "main.pro"

/*
 * Support for Cygwin binary/text mode filesystems.
 * Peter A. Castro <doctor@fruitbat.org>
 *
 * This deserves some explanation, because it uses Cygwin specific
 * runtime functions.
 *
 * Cygwin supports the notion of binary or text mode access to files
 * based on the mount attributes of the filesystem.  If a file is on
 * a binary mounted filesystem, you get exactly what's in the file, CRLF's
 * and all.  If it's on a text mounted filesystem, Cygwin will strip out
 * the CRs.  This presents a problem because zsh code doesn't allow for
 * CRLF's as line terminators.  So, we must force all open files to be
 * in text mode regardless of the underlying filesystem attributes.
 * However, we only want to do this for reading, not writing as we still
 * want to write files in the mode of the filesystem.  To do this,
 * we have two options: augment all {f}open() calls to have O_TEXT added to
 * the list of file mode options, or have the Cygwin runtime do it for us.
 * I choose the latter. :)
 *
 * Cygwin's runtime provides pre-execution hooks which allow you to set
 * various attributes for the process which effect how the process functions.
 * One of these attributes controls how files are opened.  I've set
 * it up so that all files opened RDONLY will have the O_TEXT option set,
 * thus forcing line termination manipulation.  This seems to solve the
 * problem (at least the Test suite runs clean :).
 *
 * Note: this may not work in later implementations.  This will override
 * all mode options passed into open().  Cygwin (really Windows) doesn't
 * support all that much in options, so for now this is OK, but later on
 * it may not, in which case O_TEXT will have to be added to all opens calls
 * appropriately.
 *
 * This function is actually a hook in the Cygwin runtime which
 * is called before the main of a program.  Because it's part of the program
 * pre-startup, it must be located in the program main and not in a DLL.
 * It must also be made an export so the linker resolves this function to
 * our code instead of the default Cygwin stub routine.
 */

/**/
#ifdef __CYGWIN__
/**/
mod_export void
cygwin_premain0 (int argc, char **argv, void *myself)
{
    static struct __cygwin_perfile pf[] =
    {
        {"", O_RDONLY | O_TEXT},
        {NULL, 0}
    };
    cygwin_internal (CW_PERFILE, pf);
}
/**/
#endif /* __CYGWIN__ */

/**/
int
main(int argc, char **argv)
{
    return (zsh_main(argc, argv));
}
